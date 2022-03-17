import os
import torch
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)
from tqdm.notebook import tqdm
import importlib
import torch.nn as nn
import matplotlib.pyplot as plt

Axes3D = importlib.import_module('mpl_toolkits.mplot3d').Axes3D
from configs import cfg
from configs import config

class Model3DNet(nn.Module):
    def __init__(self,
                 model_path_list=cfg["model_path"],
                 num_samples=6000,
                 num_hidden_feat=256,
                 num_ico_vert=2562,
                 fps_num=8,
                 device=config["device"]):
        super().__init__()

        # create training models
        # src_mesh = ico_sphere(4, device)
        # self.deform_verts = torch.full(src_mesh.verts_packed().shape, 0.0, device=device, requires_grad=True)

        # load the 3d model
        self.model_path_list = model_path_list
        self.device = device
        self.fps_num = fps_num
        self.meshes, self.scale_list, self.center_list, self.fps_list = self.load_models()
        self.scale_list = torch.stack(self.scale_list)[:, None]
        self.center_list = torch.stack(self.center_list)
        self.fps_points = torch.stack(self.fps_list)

        self.num_samples = num_samples
        self.num_hidden_feat = num_hidden_feat
        self.num_ico_vert = num_ico_vert

        self.subnet_vert = nn.Sequential(
            nn.Linear(num_samples * 3, self.num_hidden_feat),
            nn.ReLU(),
            nn.BatchNorm1d(self.num_hidden_feat),
            nn.Dropout()
        )

        self.subnet_normal = nn.Sequential(
            nn.Linear(num_samples * 3, self.num_hidden_feat),
            nn.ReLU(),
            nn.BatchNorm1d(self.num_hidden_feat),
            nn.Dropout()
        )

        self.subnet_vert_normal = nn.Sequential(
            nn.Linear(self.num_hidden_feat * 2, self.num_hidden_feat),
            nn.ReLU(),
            nn.BatchNorm1d(self.num_hidden_feat),
            nn.Linear(self.num_hidden_feat, self.num_hidden_feat),
            nn.ReLU()
        )

        self.subnet_decoder_samples = nn.Sequential(
            nn.Linear(self.num_hidden_feat, self.num_ico_vert),
            nn.ReLU(),
            nn.Linear(self.num_ico_vert, self.num_ico_vert * 3)
        )

        self.subnet_decoder_scale = nn.Sequential(
            nn.Linear(self.num_hidden_feat, self.num_hidden_feat // 2),
            nn.ReLU(),
            nn.Linear(self.num_hidden_feat // 2, 1)
        )

        self.subnet_decoder_center = nn.Sequential(
            nn.Linear(self.num_hidden_feat, self.num_hidden_feat // 2),
            nn.ReLU(),
            nn.Linear(self.num_hidden_feat // 2, 3)
        )

        if device is not None:
            self.to(device)

        # init the weights and bias
        self.apply(Model3DNet.initialize_parameters)

    @staticmethod
    def initialize_parameters(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias.data, 0)

    def forward(self):
        x = self.forward_encoder(self.meshes)
        samples, scales, centers = self.forward_decoder(x)

        return samples, scales, centers

    def forward_encoder(self):
        # sample points from mehses
        samples, normals = sample_points_from_meshes(self.meshes, self.num_samples, return_normals=True)

        # network
        B, _, _ = samples.shape
        x1 = self.subnet_vert(samples.reshape(B, -1))
        x2 = self.subnet_normal(normals.reshape(B, -1))

        x = self.subnet_vert_normal(torch.cat([x1, x2], dim=-1))

        return x

    def forward_decoder(self, x):
        B, _ = x.shape
        scales = self.subnet_decoder_scale(x)
        centers = self.subnet_decoder_center(x)
        x = self.subnet_decoder_samples(x)
        x = 2 * torch.sigmoid(x) - 1
        return x.reshape(B, -1, 3), scales, centers

    @staticmethod
    def load_one_model(model_path, fps_num=8, device=None):
        verts, faces, aux = load_obj(model_path)

        if device is not None:
            faces_idx = faces.verts_idx.to(device)
            verts = verts.to(device)

        center = verts.mean(0)

        fps_points = Model3DNet.get_fps_points(verts, center, fps_num=fps_num)
        verts = verts - center
        scale = max(verts.abs().max(0)[0])
        verts = verts / scale

        return verts, faces_idx, scale, center, fps_points


    @staticmethod
    def get_fps_points(model_3d_points, model_center_3d_point, fps_num=8):
        """
        Args:
            model_3d_points:  3D points of model in object coordinate  (numpy)
            model_center_3d_point: 3D points of model center in object coordinate  (numpy)
            fps_num: default 8

        Returns:

        """
        fps_3d_points = model_center_3d_point[None, :]

        for _ in range(fps_num):
            index = ((fps_3d_points[:, None, :] - model_3d_points[None, :, :])**2).sum(dim=-1).sum(dim=0).argmax(dim=-1)
            fps_3d_points = torch.cat([fps_3d_points, model_3d_points[index:index+1, :]], dim=0)

        return fps_3d_points

    def load_models(self):
        verts_faces_list = [Model3DNet.load_one_model(p, self.fps_num, self.device) for p in self.model_path_list]
        verts_list = [vf[0] for vf in verts_faces_list]
        faces_list = [vf[1] for vf in verts_faces_list]
        scale_list = [vf[2] for vf in verts_faces_list]
        center_list = [vf[3] for vf in verts_faces_list]
        fps_points_list = [vf[4] for vf in verts_faces_list]

        return Meshes(verts=verts_list, faces=faces_list), scale_list, center_list, fps_points_list

    def get_loss(self, deform_verts_list,
                 pre_scales,
                 pre_centers,
                 num_samples=2000,
                 w_scales=10.0,
                 w_centers=10.0,
                 w_chamfer=1.0,  # Weight for the chamfer loss
                 w_edge=1.0,  # Weight for mesh edge loss
                 w_normal=0.01,  # Weight for mesh normal consistency
                 w_laplacian=0.1  # Weight for mesh laplacian smoothing
                 ):
        B = deform_verts_list.shape[0]
        # add scale and center loss
        losses = [
            nn.L1Loss()(pre_scales, self.scale_list) * w_scales,
            nn.L1Loss()(pre_centers, self.center_list) * w_centers
        ]
        for i in range(B):
            deform_verts = deform_verts_list[i]

            # Deform the mesh
            src_mesh = ico_sphere(4, device)
            new_src_mesh = src_mesh.offset_verts(deform_verts)

            # We sample 10k points from the surface of each mesh
            sample_trg = sample_points_from_meshes(self.meshes[i], num_samples)
            sample_src = sample_points_from_meshes(new_src_mesh, num_samples)

            # We compare the two sets of pointclouds by computing (a) the chamfer loss
            loss_chamfer, _ = chamfer_distance(sample_trg, sample_src)

            # and (b) the edge length of the predicted mesh
            loss_edge = mesh_edge_loss(new_src_mesh)

            # mesh normal consistency
            loss_normal = mesh_normal_consistency(new_src_mesh)

            # mesh laplacian smoothing
            loss_laplacian = mesh_laplacian_smoothing(new_src_mesh, method="uniform")

            # Weighted sum of the losses
            losses.append(
                loss_chamfer * w_chamfer + loss_edge * w_edge + loss_normal * w_normal + loss_laplacian * w_laplacian)

        return sum(losses)

    @staticmethod
    def plot_pointcloud(mesh, title="plt"):
        # Sample points uniformly from the surface of the mesh.
        points = sample_points_from_meshes(mesh, 5000)
        x, y, z = points.clone().detach().cpu().squeeze().unbind(1)
        fig = plt.figure(figsize=(5, 5))
        ax = Axes3D(fig)
        ax.scatter3D(x, z, -y)
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_zlabel('y')
        ax.set_title(title)
        ax.view_init(190, 30)
        fig = plt.gcf()
        plt.savefig("./img/{}".format(title))
        plt.show()

    def save_models(self, deform_verts_list,
                    scales,
                    centers,
                    model_save_folder="./model_pre",
                    tag=0):
        B = deform_verts_list.shape[0]
        for i in range(B):
            deform_verts = deform_verts_list[i]
            scale = scales[i]
            center = centers[i]

            # Deform the mesh
            src_mesh = ico_sphere(4, device)
            new_src_mesh = src_mesh.offset_verts(deform_verts)

            # Fetch the verts and faces of the final predicted mesh
            verts_pre, faces_pre = new_src_mesh.get_mesh_verts_faces(0)

            # Scale normalize back to the original target size
            verts_pre = verts_pre * scale + center

            # Store the predicted mesh using save_obj
            obj_pre = os.path.join(model_save_folder, 'model_{}_{}.obj'.format(tag, i))
            save_obj(obj_pre, verts_pre, faces_pre)


if __name__ == "__main__":
    # Set the device
    if torch.cuda.is_available():
        device = torch.device("cuda:2")
        print(device)
    else:
        device = torch.device("cpu")
        print("WARNING: CPU only, this will be slow!")

    base_path = "../../kitti/model_kitti/"
    model = ["car_001.obj", "pedestrian_001.obj",
             "tram_001.obj", "truck_001.obj", "van_001.obj"]
    model_list = []
    for m in model:
        model_list.append(os.path.join(base_path, m))

    B = len(model_list)
    net = Model3DNet(model_path_list=model_list, device=device)
    net.train()

    optimizer = torch.optim.AdamW(net.parameters())
    # Number of optimization steps
    Niter = 200000

    plot_period = 250
    loop = tqdm(range(Niter))

    for l in loop:
        # Initialize optimizer
        optimizer.zero_grad()

    deform_verts_list, scales, centers = net()

    loss = net.get_loss(deform_verts_list, scales, centers)

    if l % plot_period == 0:
        net.save_models(deform_verts_list, scales, centers, tag=l)

    # Print the losses
    loop.set_description('total_loss = %.6f' % loss)
    print(l, '  total_loss = %.6f' % loss)

    # Optimization step
    loss.backward()
    optimizer.step()
