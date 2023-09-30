import numpy as np

from data import cornell_data
import torch
from grasp_model import MPA_SegmentationNetwork
import torch.optim as optim
from Training import train,evaluation,validate
import os



epochs=30

def get_dataset(dataset_name):
    if dataset_name == 'cornell':
        from data.cornell_data import CornellDataset
        return CornellDataset

    else:
        raise NotImplementedError('Dataset Type {} is Not implemented'.format(dataset_name))

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Dataset = get_dataset("cornell")

    train_dataset = Dataset('dataset', start=0.0, end=1.0, ds_rotate=0.0,
                            random_rotate=True, random_zoom=True,
                            include_depth=False, include_rgb=True)
    train_data = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=8
    )
    val_dataset = Dataset('dataset', start=0.5, end=1.0, ds_rotate=0.0,
                          random_rotate=True, random_zoom=False,
                          include_depth=False, include_rgb=True)
    val_data = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=8
    )

    #model = MPA_SegmentationNetwork(layers=[2,2,1,1],embed_dims=[64,128,256,512],
                                   #mlp_ratios=(4,4,4,4),num_classes=1)
    model = torch.load('Weights/grasp_model.pth')
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=2,T_mult=2,eta_min=1e-6,verbose=True)
    best_iou=0.0
    best_loss=10000
    for epoch in range(epochs):
        print()
        lr_scheduler.step()


        print('Beginning Epoch {:02d}'.format(epoch))
        train_results = train(epoch, model, device, train_data, optimizer, batches_per_epoch=24, vis=True)


        print('Validating...')
        test_results = validate(model, device, val_data, 101)
        print('%d/%d = %f' % (test_results['correct'], test_results['correct'] + test_results['failed'],
                                     test_results['correct'] / (test_results['correct'] + test_results['failed'])))
        actual_loss = test_results['loss']
        iou = test_results['correct'] / (test_results['correct'] + test_results['failed'])
        if  iou >= best_iou :
            print(f'Validation IoU increased ({best_iou:.4f} --> '
                                        f'{iou:.4f}).  'f'Saving model ...')
            torch.save(model, os.path.join('Weights', 'model.pth' ))
            #torch.save(model.state_dict(), os.path.join('Weights', 'model.pth' % (epoch, iou)))
            best_iou = iou

        if actual_loss < best_loss:
            print(f'Validation Loss decreased ({best_loss:.4f} --> '
                  f'{actual_loss:.4f}).  'f'Saving model ...')
            torch.save(model, os.path.join('Weights', 'grasp_model.pth'))
            # torch.save(model.state_dict(), os.path.join('Weights', 'model.pth' % (epoch, iou)))
            best_loss = actual_loss


