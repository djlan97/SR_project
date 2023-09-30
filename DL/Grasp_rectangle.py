import matplotlib.pyplot as plt
import torch.utils.data
from grasp_model import MPA_SegmentationNetwork
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

    model = torch.load('Weights/grasp_model.pth')
    # net_ggcnn = torch.load('./output/models/211112_1458_/epoch_30_iou_0.75')
    device = torch.device("cuda:0")
    Dataset = get_dataset("cornell")

    val_dataset = Dataset('dataset', start=0.0, end=0.1, ds_rotate=0.0,
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
        fig = plt.figure(figsize=(20, 10))
        # ax = fig.add_subplot(5, 5, 1)
        # while batch_idx < 100:
        '''
        for id,(x, y, didx, rot, zoom_factor) in enumerate( val_data):
                # batch_idx += 1
                if id>24:
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
                # print(rgb_img)
                ax = fig.add_subplot(5, 5, id+1)
                ax.imshow(rgb_img)
                ax.axis('off')
                for g in gs_1:
                    g.plot(ax)
        plt.show()
        '''

        for id,(x, y, didx, rot, zoom_factor) in enumerate( val_data):
            if id > 24:
                break
            q_gt,cos_gt,sin_gt,w_gt = y
            q_gt,ang_gt,w_gt = post_process_output(q_gt,cos_gt,sin_gt,w_gt)
            gs_1 = detect_grasps(q_gt, ang_gt, width_img=w_gt, no_grasps=1)
            rgb_img = val_dataset.get_rgb(didx, rot, zoom_factor, normalise=False)
            # print(rgb_img)
            ax = fig.add_subplot(5, 5, id + 1)
            ax.imshow(rgb_img)
            ax.axis('off')
            for g in gs_1:
                g.plot(ax)
        plt.show()



