
from trainer import LOTClassTrainer
import argparse
import os
import multiprocessing


def main():
    parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Dataset
    parser.add_argument('--dataset_dir', default='datasets/agnews/', help='dataset directory')

    # --------------Hyperparameters for representation learning-------------------
    parser.add_argument('--template_count', type=int, default=12)

    # parser.add_argument('--pre_template_0', type=str, default="[Category: *mask*]")
    # parser.add_argument('--post_template_0', type=str, default="")
    # parser.add_argument('--pre_template_1', type=str, default="[Topic: *mask*]")
    # parser.add_argument('--post_template_1', type=str, default="")
    # parser.add_argument('--pre_template_2', type=str, default="The category of '")
    # parser.add_argument('--post_template_2', type=str, default="' is *mask*.")
    # parser.add_argument('--pre_template_3', type=str, default="The type of '")
    # parser.add_argument('--post_template_3', type=str, default="' is *mask*.")

    parser.add_argument('--pre_template_0', type=str, default="The topic of '")
    parser.add_argument('--post_template_0', type=str, default="' is *mask*.")
    parser.add_argument('--pre_template_1', type=str, default="[Topic: *mask*]")
    parser.add_argument('--post_template_1', type=str, default="")
    parser.add_argument('--pre_template_2', type=str, default="The category of '")
    parser.add_argument('--post_template_2', type=str, default="' is *mask*.")
    parser.add_argument('--pre_template_3', type=str, default="The topic of '")
    parser.add_argument('--post_template_3', type=str, default="' is about *mask*.")
    parser.add_argument('--pre_template_4', type=str, default="The category of '")
    parser.add_argument('--post_template_4', type=str, default="' is about *mask*.")
    parser.add_argument('--pre_template_5', type=str, default="A *mask* news:")
    parser.add_argument('--post_template_5', type=str, default="")
    parser.add_argument('--pre_template_6', type=str, default="*mask* news:")
    parser.add_argument('--post_template_6', type=str, default="")
    parser.add_argument('--pre_template_7', type=str, default="[Category: *mask*]")
    parser.add_argument('--post_template_7', type=str, default="")
    parser.add_argument('--pre_template_8', type=str, default="Following is a *mask* news:")
    parser.add_argument('--post_template_8', type=str, default="")
    parser.add_argument('--pre_template_9', type=str, default="The news '")
    parser.add_argument('--post_template_9', type=str, default="' is about *mask*.")
    parser.add_argument('--pre_template_10', type=str, default="The type of '")
    parser.add_argument('--post_template_10', type=str, default="' is *mask*.")
    parser.add_argument('--pre_template_11', type=str, default="The type of '")
    parser.add_argument('--post_template_11', type=str, default="' is about *mask*.")

    parser.add_argument('--accum_steps', type=int, default=20)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--cl_loss_weight', type=float, default=0.99)
    parser.add_argument('--mlm_loss_weight', type=float, default=0.01)
    parser.add_argument('--prompt_early_stop', type=int, default=2)
    parser.add_argument('--label_name_aug_count', type=int, default=3)
    parser.add_argument('--save_prompt_step', type=int, default=250)
    parser.add_argument('--label_name_mode', type=str, default="aug")   # "raw" or "aug"
    parser.add_argument('--train_batch_size', type=int, default=6,
                        help='batch size per GPU for training')
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--eval_batch_size', type=int, default=128,
                        help='batch size per GPU for evaluation; bigger batch size makes training faster')

    #-----------------Hyperparameters for classifier selftraining----------------------
    parser.add_argument('--classifier_lr', type=float, default=5e-5)
    parser.add_argument('--classifier_epoch', type=int, default=20)
    parser.add_argument('--prompt_thresh', type=float, default=0.6)
    parser.add_argument('--classifier_thresh', type=float, default=0.9995)
    parser.add_argument('--pseudo_label_mode', type=str, default="hard")
    parser.add_argument('--label_embed_mode', type=str, default="seperate")
    parser.add_argument('--classifier_eval_batch_size', type=int, default=256,
                        help='batch size per GPU for evaluation; bigger batch size makes training faster')
    parser.add_argument('--classifier_train_batch_size', type=int, default=24,
                        help='batch size per GPU for training')
    parser.add_argument('--classifier_accum_steps', type=int, default=5,
                        help='gradient accumulation steps during training')
    parser.add_argument('--max_len', type=int, default=512,
                        help='length that documents are padded/truncated to')
    parser.add_argument('--self_training_max_time', type=float, default=10,
                        help='self training epochs; 1-5 usually is good depending on dataset size (smaller dataset needs more epochs)')
    parser.add_argument('--train_per', type=float, default=0.9,
                        help='self training epochs; 1-5 usually is good depending on dataset size (smaller dataset needs more epochs)')
    parser.add_argument('--save_classifier_step', type=float, default=0)

    # -------------Hypeparameters do not need change---------------
    parser.add_argument('--label_names_file', default='label_names.txt',
                        help='file containing label names (under dataset directory)')
    parser.add_argument('--label_names_aug_file', default='label_names_aug.txt',
                        help='file containing label names (under dataset directory)')
    parser.add_argument('--train_file', default='train.txt',
                        help='unlabeled text corpus for training (under dataset directory); one document per line')
    parser.add_argument('--train_label_file', default='train_labels.txt',
                        help='train corpus ground truth label')
    parser.add_argument('--test_file', default='test.txt',
                        help='test corpus to conduct model predictions (under dataset directory); one document per line')
    parser.add_argument('--test_label_file', default='test_labels.txt',
                        help='test corpus ground truth label; if provided, model will be evaluated during self-training')
    parser.add_argument('--final_model', default='final_model.pt',
                        help='the name of the final classification model to save to')
    parser.add_argument('--final_model_classifier', default='final_model_classifier.pt',
                        help='the name of the final classification model to save to')
    parser.add_argument('--out_file', default='out.txt',
                        help='model predictions on the test corpus if provided')
    parser.add_argument('--top_pred_num', type=int, default=50,
                        help='language model MLM top prediction cutoff')
    parser.add_argument('--category_vocab_size', type=int, default=100,
                        help='category vocabulary size for each class')
    parser.add_argument('--gpus', default=1, type=int,
                        help='number of gpus to use')

    args = parser.parse_args()
    print(args)
    trainer = LOTClassTrainer(args)

    # ??????label name ???????????? label_names_aug.txt
    trainer.prepare_label_name(mode=args.label_name_mode, label_name_count_each_cat=args.label_name_aug_count,
                               top_pred_num=args.top_pred_num, category_vocab_size=args.category_vocab_size)

    # p1 = multiprocessing.Process(target=trainer.label_name_augmentation, args=(3, args.top_pred_num, args.category_vocab_size))
    # p1.start()
    # p1.join()

    # ???????????????????????????BERT
    # trainer.save_model(file_name="final_model.pt")

    # ??????Prompt??????
    # trainer.cl_train_4(rank=0, lr=args.lr, epoch=args.epoch)

    # ??????Prompt??????
    # trainer.write_results(loader_name=args.final_model, out_file=args.out_file)

    # ????????????ensemble???representation learning
    for id in range(args.template_count):
        trainer.cl_train_multi_template(rank=0, lr=args.lr, epoch=args.epoch, template_id=id)
        # ??????????????????????????????
        # p = multiprocessing.Process(target=trainer.cl_train_multi_template, args=(0, args.lr, args.epoch, id))
        # p.start()
        # p.join()

    trainer.write_results_multi_template(out_file=args.out_file, template_count=args.template_count, mode="testset_template", label_mode=args.label_name_mode)

    # ??????classifier
    # trainer.train_classifier_selftrain()

    # ?????????BERT finetuning??????classifier????????????template?????????
    trainer.train_classifier_selftrain_without_template()

    # ??????classifier with multi template (??????confidence factor ?????????template)
    trainer.train_classifier_selftrain_multi_template()

    # supervised learning - check finetune ?????????
    # trainer.supervised_classifier_finetuning()

    # ??????checkpoint???????????????
    # trainer.check_point_eval()

    # 5???template????????????classifier????????????????????????emsemble
    # trainer.train_classifier_selftrain_multi_template_multi_classifier(template_count=args.template_count)

    # ??????classifier
    # trainer.eval_classifier(out_file="out.txt")


if __name__ == "__main__":
    main()
    