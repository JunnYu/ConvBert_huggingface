### 转化discriminators权重

- 修改convert.py中的tf_checkpoint_path

- python convert.py

### 测试

- python test.py

- 预计输出：
  ```python
   cuda下的测试：
   use_cuda_kernel=True时:
   small max difference is : tensor(3.3379e-06)
   small mean difference is : tensor(2.9952e-07)
   medium-small max difference is : tensor(3.4571e-06)
   medium-small mean difference is : tensor(2.6056e-07)
   base max difference is : tensor(5.7220e-06)
   base mean difference is : tensor(1.9045e-07)

   use_cuda_kernel=False时:
   small max difference is : tensor(4.5300e-06)
   small mean difference is : tensor(3.3393e-07)
   medium-small max difference is : tensor(3.2187e-06)
   medium-small mean difference is : tensor(2.6430e-07)
   base max difference is : tensor(5.7220e-06)
   base mean difference is : tensor(1.9694e-07)
  ```
### 其他
- 通过设置config.use_cuda_kernal变量，决定是否使用fairseq的cuda版本的dynamicconv_layer。
- dynamicconv_layer安装：cd convbert_huggingface/dynamicconv_layer & python setup.py install
- transformers==3.5.1