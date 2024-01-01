import os.path

import matplotlib.pyplot as plt
import torch
import numpy as np


def plotSlice(vol, slice, case, config, seg_vol=None, name=None, path=None):
    fig, axes = plt.subplots(nrows=2, ncols=3, gridspec_kw={'wspace': 0, 'hspace': 0})
    for ax, (i, bval) in zip(axes.flat, enumerate(config['b_vector'])):
        im = ax.imshow(vol[i, :, :, slice].squeeze(), cmap='gray', vmax=torch.max(vol[0, :, :, slice]).item(), vmin=0)
        if seg_vol is not None:
            ax.contour(seg_vol[i, :, :, slice].squeeze(), linewidths=1, colors='red', linestyles='solid')
        # ax.set_title("b = " + str(bval.item()) + r'$ \frac{s}{mm^2}$')
        ax.axis('off')

    # plt.tight_layout()
    if path is None:
        fig.show()
    else:
        path_all = os.path.join(path, f'Case {case[4:]} {name}.pdf')
        fig.savefig(path_all, format='pdf')
    plt.close()

    for (i, bval) in enumerate(config['b_vector']):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        im = ax.imshow(vol[i, :, :, slice].squeeze(), cmap='gray', vmax=torch.max(vol[0, :, :, slice]).item(), vmin=0)
        if seg_vol is not None:
            ax.contour(seg_vol[i, :, :, slice].squeeze(), linewidths=1, colors='red', linestyles='solid')
            # ax.contour(seg_vol[:, :, slice].squeeze(), linewidths=1, colors='red', linestyles='solid')
            # tmp_seg = seg_vol[i,:, :, slice].squeeze()
            # tmp_seg[tmp_seg == 0] = torch.nan
            # ax.imshow(tmp_seg, cmap='jet', alpha = 0.5)
        # ax.set_title("b = " + str(bval.item()) + r'$ \frac{s}{mm^2}$')
        ax.axis('off')

        # plt.tight_layout()
        if path is None:
            fig.show()
        else:
            path_b = os.path.join(path, f'Case {case[4:]} {name}_b{i}.jpeg')
            fig.savefig(path_b, format='jpeg')
        plt.close()


def plotParamMap(ADCmap, Slice, case, seg_vol=None, path=None):
    fig = plt.figure()
    plt.imshow(ADCmap[:, :, Slice], cmap='gray')
    if seg_vol is not None:
        plt.contour(seg_vol[0, :, :, Slice].squeeze(), linewidths=1, colors='red', linestyles='solid')
    plt.axis('off')
    # plt.colorbar()
    # plt.title(f'{title}')
    if path is None:
        plt.show()
    else:
        path = os.path.join(path, f'Case {case[4:]} ADC_0.jpeg')
        plt.savefig(path)
    plt.close()


def plotmask(mask, Slice, path=None):
    fig = plt.figure()
    plt.imshow(mask[:, :, Slice], cmap='gray')
    plt.axis('off')
    plt.title(f'Mask')
    if path is None:
        plt.show()
    else:
        plt.savefig(path)
    plt.close()


def PlotMovingFixedADC(moving_img_history, model_recon_history, ADC_history, ConvIter, slice, case, path=None):
    fig, axes = plt.subplots(nrows=3, ncols=5, gridspec_kw={'wspace': 0, 'hspace': 0})
    iter_idx = [0 + i * ConvIter // 4 for i in range(5)]
    for i in range(5):
        axes[0, i].imshow(moving_img_history[iter_idx[i]][0, :, :, slice].squeeze(), cmap='gray')
        axes[1, i].imshow(ADC_history[iter_idx[i]][:, :, slice].squeeze(), cmap='gray')
        axes[2, i].imshow(model_recon_history[iter_idx[i]][0, :, :, slice].squeeze(), cmap='gray')
        axes[0, i].set_title(f'Iter #{iter_idx[i]}')
        for j in range(3):
            axes[j, i].set_xticks([])
            axes[j, i].set_yticks([])
    axes[0, 0].set_ylabel('Moving Image\n $b=0\ s/mm^2$')
    axes[1, 0].set_ylabel('ADC Map\n $mm^2/s$')
    axes[2, 0].set_ylabel('Fixed Image\n $b=0\ s/mm^2$')
    plt.tight_layout()
    if path is None:
        fig.show()
    else:
        path = os.path.join(path, f'Case {case[4:]} Summery.pdf')
        fig.savefig(path, format='pdf')
    plt.close()


def PlotROI(s0_IRLS_history, ADC_IRLS_history, ROI_history, ConvIter, config, case, path=None):
    iter_idx = [0 + i * ConvIter // 4 for i in range(5)]
    fig, ax = plt.subplots()
    b_vector = config['b_vector'].detach().cpu()
    for i in iter_idx:
        plt.plot(b_vector, torch.log(s0_IRLS_history[i]) - b_vector * ADC_IRLS_history[i], '--', label=f'iter # {i}')
        plt.scatter(b_vector, torch.log(torch.mean(ROI_history[i], 1)),
                    label='_nolegend_')

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_aspect(abs(x1 - x0) / abs(y1 - y0))
    ax.grid(b=True, which='major', color='k', linestyle='--')
    plt.xlabel('B-value $(s/mm^2)$')
    plt.ylabel('$log(Signal)$')
    plt.legend()
    plt.savefig(os.path.join(path, f'ROI_iter_{case}_log.pdf'), format='pdf')
    plt.close(fig)

    for i in iter_idx:
        fig, ax = plt.subplots()
        b_vector = config['b_vector'].detach().cpu()
        plt.plot(b_vector, torch.exp(- b_vector * ADC_IRLS_history[i]), '--', label=f'iter # {i}')
        plt.scatter(b_vector, torch.mean(ROI_history[i], 1) / torch.mean(ROI_history[i], 1)[0],
                    label='_nolegend_')
        plt.xlim((0, 600))
        plt.ylim((0, 1))
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ax.set_aspect(abs(x1 - x0) / abs(y1 - y0))
        ax.grid(b=True, which='major', color='k', linestyle='--')
        plt.xlabel('B-value $(s/mm^2)$')
        plt.ylabel('$S/S_0$')
        plt.savefig(os.path.join(path, f'ROI_iter_{case}_{i}.pdf'), format='pdf')
        plt.close(fig)

    for i in iter_idx:
        fig, ax = plt.subplots()
        b_vector = config['b_vector'].detach().cpu()
        plt.plot(b_vector, - b_vector * ADC_IRLS_history[i], '--', label=f'iter # {i}')
        plt.scatter(b_vector, torch.log(torch.mean(ROI_history[i], 1) / torch.mean(ROI_history[i], 1)[0]),
                    label='_nolegend_')
        plt.xlim((0, 600))
        plt.ylim((0, 1))
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ax.set_aspect(abs(x1 - x0) / abs(y1 - y0))
        ax.grid(b=True, which='major', color='k', linestyle='--')
        plt.xlabel('B-value $(s/mm^2)$')
        plt.ylabel('$log(S/S_0)$')
        plt.savefig(os.path.join(path, f'ROI_iter_{case}_{i}_log.pdf'), format='pdf')
        plt.close(fig)
    # fig, ax = plt.subplots()
    # b_vector = config['b_vector'].detach().cpu()
    # for i in iter_idx:
    #     plt.plot(b_vector, torch.exp( - b_vector * ADC_IRLS_history[i]), '--', label=f'iter # {i}')
    #     plt.scatter(b_vector, torch.mean(ROI_history[i], 1)/torch.mean(ROI_history[i], 1)[0],
    #                 label='_nolegend_')
    #
    # x0, x1 = ax.get_xlim()
    # y0, y1 = ax.get_ylim()
    # ax.set_aspect(abs(x1 - x0) / abs(y1 - y0))
    # ax.grid(b=True, which='major', color='k', linestyle='--')
    # plt.xlabel('B-value $(s/mm^2)$')
    # plt.ylabel('$(S/S_0)$')
    # plt.legend()
    # plt.savefig(os.path.join(path, f'ROI_iter_{case}.pdf'), format='pdf')
    # plt.close(fig)


def PlotLossR_2(case, loss, R_2, path):
    fig, axes = plt.subplots(nrows=1, ncols=2, gridspec_kw={'wspace': 0, 'hspace': 0})
    axes[0].plot(R_2)
    axes[0].set_xlabel('iteration')
    axes[0].set_ylabel('R^2')
    x0, x1 = axes[0].get_xlim()
    y0, y1 = axes[0].get_ylim()
    axes[0].set_aspect(abs(x1 - x0) / abs(y1 - y0))
    axes[0].grid(b=True, which='major', color='k', linestyle='--')
    axes[1].plot(loss)
    axes[1].set_xlabel('iteration')
    axes[1].set_ylabel('loss')
    x0, x1 = axes[1].get_xlim()
    y0, y1 = axes[1].get_ylim()
    axes[1].set_aspect(abs(x1 - x0) / abs(y1 - y0))
    axes[1].grid(b=True, which='major', color='k', linestyle='--')

    plt.savefig(os.path.join(path, f'loss_r2 case{case}.pdf'), format='pdf')
    plt.close(fig)


def plotReg(vol, slice, case, config, name=None, path=None):
    fig, axes = plt.subplots(nrows=2, ncols=3, gridspec_kw={'wspace': 0, 'hspace': 0})
    for ax in axes.flat:
        ax.axis('off')

    for ax, (i, bval) in zip(axes.flat, enumerate(config['b_vector'])):
        b_n = vol[i, :, :, slice].squeeze()
        b_n = 1. / (b_n.max() - b_n.min()) * b_n + 1. * b_n.min() / (b_n.min() - b_n.max())
        b_0 = vol[0, :, :, slice].squeeze()
        b_0 = 1. / (b_0.max() - b_0.min()) * b_0 + 1. * b_0.min() / (b_0.min() - b_0.max())

        a = b_0.cpu().numpy()
        b = b_n.cpu().numpy()
        C = np.dstack((a, b * 0.5 + a * 0.5, a))
        im = ax.imshow(C)
    # plt.tight_layout()
    if path is None:
        fig.show()
    else:
        path_all = os.path.join(path, f'Case {case[4:]} {name}.pdf')
        fig.savefig(path_all, format='pdf')
    plt.close()


def PlotRegIter(img_history, ConvIter, slice, case, config, name=None, path=None):
    # fig, axes = plt.subplots(nrows=3, ncols=5, gridspec_kw={'wspace': 0, 'hspace': 0})
    iter_idx = [0 + i * ConvIter // 4 for i in range(5)]
    for i in range(5):
        plotReg(img_history[iter_idx[i]], slice, case, config, f'{name}_{iter_idx[i]}', path)

# plot phi:
# from PIL import Image
#
# r = pred[0][0,:,:,:,seg_slice[0]].detach().cpu().squeeze().numpy()
# r += r.min()
# r *= 255.0/r.max()
#
# g = pred[0][1,:,:,:,seg_slice[0]].detach().cpu().squeeze().numpy()
# g += g.min()
# g *= 255.0/g.max()
#
# b = pred[0][2,:,:,:,seg_slice[0]].detach().cpu().squeeze().numpy()
# b += b.min()
# b *= 255.0/b.max()
#
# rgbArray = np.zeros((96,96,3), 'uint8')
# rgbArray[..., 0] = r
# rgbArray[..., 1] = g
# rgbArray[..., 2] = b
#
# img = Image.fromarray(rgbArray)
# img.save('phi.jpeg', dpi =(800,800))
