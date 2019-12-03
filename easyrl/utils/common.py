import json
import random
import shutil
from pathlib import Path

import cv2
import git
import numpy as np
import torch
from easyrl.utils.rl_logger import logger


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_traj(traj, save_dir, start_idx=0):
    if isinstance(save_dir, str):
        save_dir = Path(save_dir)
    save_state = traj[0].state is not None
    ob_is_state = len(np.array(traj[0].ob[0]).shape) <= 1
    infos = traj.infos
    actions = traj.actions
    tsps = traj.steps_til_done.copy().tolist()
    folder_idx = start_idx
    for ei in range(traj.num_envs):
        ei_save_dir = save_dir.joinpath('{:06d}'.format(folder_idx))
        ei_render_imgs = []
        for t in range(tsps[ei]):
            img_t = infos[t][ei]['render_image']
            ei_render_imgs.append(img_t)
        img_folder = ei_save_dir.joinpath('render_imgs')
        save_images(ei_render_imgs, img_folder)

        if ob_is_state:
            ob_file = ei_save_dir.joinpath('obs.json')
            save_to_json(traj.obs[:tsps[ei], ei].tolist(),
                         ob_file)
        else:
            ob_folder = ei_save_dir.joinpath('obs')
            save_images(traj.obs[:tsps[ei], ei], ob_folder)
        action_file = ei_save_dir.joinpath('actions.json')
        save_to_json(actions[:tsps[ei], ei].tolist(),
                     action_file)
        if save_state:
            state_file = ei_save_dir.joinpath('states.json')
            save_to_json(traj.states[:tsps[ei], ei].tolist(),
                         state_file)
        folder_idx += 1
    return folder_idx


def save_images(images, save_dir):
    if isinstance(save_dir, str):
        save_dir = Path(save_dir)
    if save_dir.exists():
        shutil.rmtree(save_dir, ignore_errors=True)
    Path.mkdir(save_dir, parents=True)
    for i in range(len(images)):
        img = images[i]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_file_name = save_dir.joinpath('{:06d}.png'.format(i))
        cv2.imwrite(img_file_name.as_posix(), img)


def save_to_json(data, file_name):
    if isinstance(file_name, str):
        file_name = Path(file_name)
    if not file_name.parent.exists():
        Path.mkdir(file_name.parent, parents=True)
    with file_name.open('w') as f:
        json.dump(data, f, indent=2)


def tile_images(img_nhwc):
    """
    Tile N images into one big PxQ image
    (P,Q) are chosen to be as close as possible, and if N
    is square, then P=Q.

    input: img_nhwc, list or array of images, ndim=4 once turned into array
        n = batch index, h = height, w = width, c = channel
    returns:
        bigim_HWc, ndarray with ndim=3
    """
    img_nhwc = np.asarray(img_nhwc)
    N, h, w, c = img_nhwc.shape
    H = int(np.ceil(np.sqrt(N)))
    W = int(np.ceil(float(N) / H))
    img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0] * 0 for _ in range(N, H * W)])
    img_HWhwc = img_nhwc.reshape(H, W, h, w, c)
    img_HhWwc = img_HWhwc.transpose(0, 2, 1, 3, 4)
    img_Hh_Ww_c = img_HhWwc.reshape(H * h, W * w, c)
    return img_Hh_Ww_c


def linear_decay_percent(epoch, total_epochs):
    return 1 - epoch / float(total_epochs)


def get_list_stats(data):
    min_data = np.amin(data)
    max_data = np.amax(data)
    mean_data = np.mean(data)
    median_data = np.median(data)
    stats = dict(
        min=min_data,
        max=max_data,
        mean=mean_data,
        median=median_data
    )
    return stats


def get_git_infos(path):
    git_info = None
    try:
        repo = git.Repo(path)
        try:
            branch_name = repo.active_branch.name
        except TypeError:
            branch_name = '[DETACHED]'
        git_info = dict(
            directory=str(path),
            code_diff=repo.git.diff(None),
            code_diff_staged=repo.git.diff('--staged'),
            commit_hash=repo.head.commit.hexsha,
            branch_name=branch_name,
        )
    except git.exc.InvalidGitRepositoryError as e:
        logger.error(f'Not a valid git repo: {path}')
    except git.exc.NoSuchPathError as e:
        logger.error(f'{path} does not exist.')
    return git_info
