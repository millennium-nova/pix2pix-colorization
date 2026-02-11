"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
import wandb
import matplotlib.pyplot as plt
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer

if __name__ == '__main__':
    opt = TrainOptions().parse()   # トレーニングオプションを取得
    dataset = create_dataset(opt)  # opt.dataset_mode とその他のオプションに基づいてデータセットを作成
    dataset_size = len(dataset)    # データセット内の画像の数を取得
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # opt.model とその他のオプションに基づいてモデルを作成
    model.setup(opt)               # 通常のセットアップ：ネットワークをロードして表示し、スケジューラを作成
    visualizer = Visualizer(opt)   # 画像とプロットを表示/保存するビジュアライザを作成
    total_iters = 0                # トレーニングイテレーションの総数
    losses_record = []             # 損失を格納するための配列

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # エポックごとの外側のループ。モデルは <epoch_count>、<epoch_count>+<save_latest_freq> で保存する
        epoch_start_time = time.time()  # エポック全体のタイマー
        iter_data_time = time.time()    # イテレーションごとのデータ読み込みタイマー
        epoch_iter = 0                  # 現在のエポック内のトレーニングイテレーションの数。各エポックごとにリセット
        visualizer.reset()              # ビジュアライザをリセット：各エポックごとに HTML に結果が少なくとも 1 回保存されるようにする
        model.update_learning_rate()    # 各エポックの最初に学習率を更新
        for i, data in enumerate(dataset):  # エポック内の内側のループ
            iter_start_time = time.time()  # イテレーションごとの計算タイマー
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # データセットからデータを取り出し、前処理を適用
            model.optimize_parameters()   # 損失関数を計算し、勾配を取得し、ネットワークの重みを更新

            if total_iters % opt.display_freq == 0:   # Visdom に画像を表示し、HTML ファイルに画像を保存
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # トレーニングの損失を表示し、ログ情報をディスクに保存
                losses = model.get_current_losses()
                losses_record.append(losses) # 損失を配列に保存、後からグラフにプロットするため
                wandb.log({"total_iters": total_iters, "train_losses": losses})
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # <save_latest_freq> イテレーションごとに最新のモデルをキャッシュ
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # <save_epoch_freq> エポックごとにモデルをキャッシュ
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))

 # 損失をグラフにプロット
x = list(range(len(losses_record)))
y = losses_record
plt.plot(x, y)
plt.title('Train Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()