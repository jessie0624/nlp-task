import torch 
import numpy as np 
import pandas as pd  
import matchzoo as mz 
print('matchzoo version', mz.__version__)

ranking_task = mz.tasks.Ranking(losses=mz.losses.RankHingeLoss())
ranking_task.metrics = [
    mz.metrics.NormalizedDiscountedCumulativeGain(k=3),
    mz.metrics.NormalizedDiscountedCumulativeGain(k=5),
    mz.metrics.MeanAveragePrecision()
]
print("`ranking_task` initialized with metrics", ranking_task.metrics)

print('data loading ...')
train_pack_raw = mz.datasets.wiki_qa.load_data('train', task=ranking_task)
dev_pack_raw = mz.datasets.wiki_qa.load_data('dev', task=ranking_task, filtered=True)
test_pack_raw = mz.datasets.wiki_qa.load_data('test', task=ranking_task, filtered=True)
print('data loaded as `train_pack_raw` `dev_pack_raw` `test_pack_raw`')


ranking_task = mz.tasks.Ranking(losses=mz.losses.RankCrossEntropyLoss(num_neg=4))
ranking_task.metrics = [
    mz.metrics.NormalizedDiscountedCumulativeGain(k=3),
    mz.metrics.NormalizedDiscountedCumulativeGain(k=5),
    mz.metrics.MeanAveragePrecision()
]

preprocessor = mz.models.DSSM.get_default_preprocessor(ngram_size=3)
train_pack_processed = preprocessor.fit_transform(train_pack_raw)
valid_pack_processed = preprocessor.transform(dev_pack_raw)
test_pack_processed = preprocessor.transform(test_pack_raw)

print(preprocessor.context)

triletter_callback = mz.dataloader.callbacks.Ngram(preprocessor, mode='aggregate')

trainset = mz.dataloader.Dataset(
    data_pack=train_pack_processed,
    mode='pair',
    batch_size=64,
    num_dup=1,
    num_neg=4,
    callbacks=[triletter_callback]
)

testset = mz.dataloader.Dataset(
    data_pack = test_pack_processed,
    batch_size=64,
    callbacks=[triletter_callback]
)

padding_callback = mz.models.DSSM.get_default_padding_callback()

trainloader = mz.dataloader.DataLoader(
    dataset=trainset,
    stage='train',
    resample=True,
    callback=padding_callback
)
testloader = mz.dataloader.DataLoader(
    dataset=testset,
    stage='dev',
    callback=padding_callback
)

model = mz.models.DSSM()
model.params['task'] = ranking_task
model.params['vocab_size'] = preprocessor.context['ngram_vocab_size']
model.params['mlp_num_layers'] = 3
model.params['mlp_num_units'] = 300 
model.params['mlp_num_fan_out'] = 128 
model.params['mlp_activation_func'] = 'relu'
model.build()

print(model, sum(p.numel() for p in model.parameters() if p.requires_grad))

optimizer = torch.optim.Adam(model.parameters())
trainer = mz.trainers.Trainer(
    model=model,
    optimizer=optimizer,
    trainloader=trainloader,
    validloader=testloader,
    validate_interval=None,
    epochs=10
)
trainer.run()