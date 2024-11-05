import os
import tqdm
import torch
import argparse
from PIL import Image
from utils.zero123_utils import init_model, predict_stage1_gradio, zero123_infer
# from utils.sam_utils import sam_init, sam_out_nosave
from utils.utils import pred_bbox, image_preprocess_nosave, gen_poses, convert_mesh_format
from elevation_estimate.estimate_wild_imgs import estimate_elev


def preprocess(predictor, raw_im, lower_contrast=False):
    raw_im.thumbnail([512, 512], Image.Resampling.LANCZOS)
    image_sam = raw_im  # sam_out_nosave(predictor, raw_im.convert("RGB"), pred_bbox(raw_im))
    input_256 = image_preprocess_nosave(image_sam, lower_contrast=lower_contrast, rescale=True)
    torch.cuda.empty_cache()
    return input_256

def stage1_run(model, device, exp_dir,
               input_im, scale, ddim_steps):
    # folder to save the stage 1 images
    stage1_dir = os.path.join(exp_dir, "stage1_8")
    os.makedirs(stage1_dir, exist_ok=True)

    # stage 1: generate 4 views at the same elevation as the input
    output_ims = predict_stage1_gradio(model, input_im, save_path=stage1_dir, adjust_set=list(range(4)) + list(range(12, 16)), device=device, ddim_steps=ddim_steps, scale=scale)
    # output_ims = predict_stage1_gradio(model, input_im, save_path=stage1_dir, adjust_set=list(range(4)), device=device, ddim_steps=ddim_steps, scale=scale)
    output_ims = output_imgs[:4]
    
    # stage 2 for the first image
    # infer 4 nearby views for an image to estimate the polar angle of the input
    stage2_steps = 50 # ddim_steps
    zero123_infer(model, exp_dir, indices=[0], device=device, ddim_steps=stage2_steps, scale=scale)
    # estimate the camera pose (elevation) of the input image.
    try:
        polar_angle = int(estimate_elev(exp_dir))
    except:
        print("Failed to estimate polar angle")
        polar_angle = 90
    print("Estimated polar angle:", polar_angle)
    gen_poses(exp_dir, polar_angle)

    # stage 1: generate another 4 views at a different elevation
    if polar_angle <= 75:
        output_ims_2 = predict_stage1_gradio(model, input_im, save_path=stage1_dir, adjust_set=list(range(4,8)), device=device, ddim_steps=ddim_steps, scale=scale)
    else:
        output_ims_2 = predict_stage1_gradio(model, input_im, save_path=stage1_dir, adjust_set=list(range(8,12)), device=device, ddim_steps=ddim_steps, scale=scale)
    torch.cuda.empty_cache()
    return 90-polar_angle, output_ims+output_ims_2
    
def stage2_run(model, device, exp_dir,
               elev, scale, stage2_steps=50):
    # stage 2 for the remaining 7 images, generate 7*4=28 views
    if 90-elev <= 75:
        zero123_infer(model, exp_dir, indices=list(range(1,8)), device=device, ddim_steps=stage2_steps, scale=scale)
    else:
        zero123_infer(model, exp_dir, indices=list(range(1,4))+list(range(8,12)), device=device, ddim_steps=stage2_steps, scale=scale)

def reconstruct(exp_dir, output_format=".ply", device_idx=0, resolution=256):
    exp_dir = os.path.abspath(exp_dir)
    main_dir_path = os.path.abspath(os.path.dirname("./"))
    os.chdir('reconstruction/')

    bash_script = f'CUDA_VISIBLE_DEVICES={device_idx} python exp_runner_generic_blender_val.py \
                    --specific_dataset_name {exp_dir} \
                    --mode export_mesh \
                    --conf confs/one2345_lod0_val_demo.conf \
                    --resolution {resolution}'
    print(bash_script)
    os.system(bash_script)
    os.chdir(main_dir_path)

    ply_path = os.path.join(exp_dir, f"mesh.ply")
    if output_format == ".ply":
        return ply_path
    if output_format not in [".obj", ".glb"]:
        print("Invalid output format, must be one of .ply, .obj, .glb")
        return ply_path
    return convert_mesh_format(exp_dir, output_format=output_format)


def predict_multiview(model_zero123, predictor, device, shape_dir, image_path, half_precision, gpu_idx):
    input_raw = Image.open(image_path)
    # preprocess the input image
    input_256 = preprocess(predictor, input_raw)
    # generate multi-view images in two stages with Zero123.
    # first stage: generate N=8 views cover 360 degree of the input shape.
    elev, stage1_imgs = stage1_run(model_zero123, device, shape_dir, input_256, scale=3, ddim_steps=75)
    # second stage: 4 local views for each of the first-stage view, resulting in N*4=32 source view images.
    stage2_run(model_zero123, device, shape_dir, elev, scale=3, stage2_steps=50)

if __name__ == "__main__":
    assert(torch.cuda.is_available())
    half_precision=False
    gpu_idx=0
    device = f"cuda:{gpu_idx}"
    # initialize the zero123 model
    models = init_model(device, 'zero123-xl.ckpt', half_precision=half_precision)
    model_zero123 = models["turncam"]
    # initialize the Segment Anything model
    predictor = None  # sam_init(gpu_idx)
    

    parent_dir = '/home/rkabra_google_com/data/objaverse_xl-holdouts/'
    uids = os.listdir(parent_dir)
    uids = [uid for uid in uids if len(uid) == 64]
    uids_completed = os.listdir('exp')
    uids = list(set(uids) - set(uids_completed))
    for uid in tqdm.tqdm(uids):
        try:
            image_path = os.path.join(parent_dir, uid, 'textured_000.png')
            shape_dir = f"./exp/{uid}"
            os.makedirs(shape_dir, exist_ok=True)

            predict_multiview(model_zero123, predictor, device, shape_dir, image_path, half_precision, gpu_idx)

            # utilize cost volume-based 3D reconstruction to generate textured 3D mesh
            mesh_path = reconstruct(shape_dir, output_format='.glb', device_idx=0, resolution=256)
            print("Mesh saved to:", mesh_path)
        except Exception:
            print("Failed uid %s" % uid)
