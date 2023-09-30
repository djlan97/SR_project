import numpy as np
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
import matplotlib.pyplot as plt
import torch.utils.data

from dataset_processing.grasp import detect_grasps,GraspRectangles
from Training import post_process_output
import cv2
import matplotlib

def get_dataset(dataset_name):
    if dataset_name == 'cornell':
        from data.cornell_data import CornellDataset
        return CornellDataset

    else:
        raise NotImplementedError('Dataset Type {} is Not implemented'.format(dataset_name))

if __name__ == '__main__':

    model = torch.load('Weights/model.pth')
    # net_ggcnn = torch.load('./output/models/211112_1458_/epoch_30_iou_0.75')
    device = torch.device("cuda:0")
    Dataset = get_dataset("cornell")

    val_dataset = Dataset('dataset', start=0.7, end=1.0, ds_rotate=0.0,
                          random_rotate=False, random_zoom=False,
                          include_depth=False, include_rgb=True)
    val_data = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=8
    )

    results = {
        'correct': 0,
        'failed': 0,
        'loss': 0,
        'losses': {

        }
    }
    ld = len(val_data)
    with torch.no_grad():
        batch_idx = 0
        # fig = plt.figure(figsize=(10, 10))
        # ax = fig.add_subplot(1, 4, 1)
        # while batch_idx < 100:
        for id,(x, y, didx, rot, zoom_factor) in enumerate( val_data):
                # batch_idx += 1

                if id > 9:
                    break

                print(id)
                print(x.shape)
                xc = x.to(device)
                yc = [yy.to(device) for yy in y]
                lossd = model.compute_loss(xc, yc)

                loss = lossd['loss']

                results['loss'] += loss.item() / ld
                for ln, l in lossd['losses'].items():
                    if ln not in results['losses']:
                        results['losses'][ln] = 0
                    results['losses'][ln] += l.item() / ld

                q_out, ang_out, w_out = post_process_output(lossd['pred']['pos'], lossd['pred']['cos'],
                                                            lossd['pred']['sin'], lossd['pred']['width'])
                gs_1 = detect_grasps(q_out, ang_out, width_img=w_out, no_grasps=1)
                rgb_img=val_dataset.get_rgb(didx, rot, zoom_factor, normalise=False)

                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(1, 4, id+1)
                ax.imshow(rgb_img)
                plt.show()
'''
                ax = fig.add_subplot(1, 4, 2)
                plot = ax.imshow(q_out, cmap="jet", vmin=0, vmax=1)
                plt.colorbar(plot)
                ax.axis('off')
                ax.set_title('q image')

                ax = fig.add_subplot(1, 4, 3)  # flag  prism jet
                plot = ax.imshow(ang_out, cmap="hsv", vmin=-np.pi / 2, vmax=np.pi / 2)
                plt.colorbar(plot)
                ax.axis('off')
                ax.set_title('angle')

                ax = fig.add_subplot(1, 4, 4)
                plot = ax.imshow(w_out, cmap='jet', vmin=-0, vmax=150)
                plt.colorbar(plot)
                ax.set_title('width')
                ax.axis('off')
                # print(rgb_img)

                plt.show()
                #plt.savefig('RGB_1_%d.pdf' % 1, bbox_inches='tight')
'''

