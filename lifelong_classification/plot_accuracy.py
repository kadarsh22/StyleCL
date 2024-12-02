import numpy as np
import matplotlib.pyplot as plt
import scipy.io


def draw_result(joint, mergan, stylecl , ewc , iter, title,iter2):
    plt.plot(iter, joint, '--k', label='Joint')
    # plt.plot(iter2, mergan, '--r', label='MeRGAN')
    plt.plot(iter, stylecl, '-g', label='StyleCL')
    plt.plot(iter, ewc, '--b', label='EWC')

    plt.xlabel("n iteration")
    plt.ylabel("Test accuracy on all tasks (%)")
    plt.legend(loc='upper left')
    plt.title(title)
    plt.savefig('conditional/'+title+".png")  # should before plt.show method
    plt.show()


def test_draw():

    joint = scipy.io.loadmat('conditional/quant_results/joint_quant_results.mat')['ACC_all'][0]
    mergan = scipy.io.loadmat('conditional/quant_results/MeRGAN_quant_results.mat')['ACC_all'][0]
    stylecl = scipy.io.loadmat('conditional/quant_results/StyleCL_quant_results.mat')['ACC_all'][0]
    ewc = scipy.io.loadmat('conditional/quant_results/EWC_quant_results.mat')['ACC_all'][0]
    iter = np.array(range(len(stylecl)))
    iter2 = np.array(range(len(mergan)))
    draw_result(joint, mergan, stylecl , ewc , iter , "cond_classification_plot", iter2)


if __name__ == '__main__':
    test_draw()