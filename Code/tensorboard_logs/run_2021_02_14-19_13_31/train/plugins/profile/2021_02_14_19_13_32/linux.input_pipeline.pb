	1[�*�M@1[�*�M@!1[�*�M@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-1[�*�M@�<,Ԛ�?1�*O ��?A鷯�?I4ڪ$��?*	��v���Y@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�l��<+�?!�vO�y"@@)m�s�p��?1�����:@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�#Di�?!�B�j-MA@)��� 4J�?1Rr�5@:Preprocessing2F
Iterator::Model��q���?!ZS��9@)gҦ�ٌ?1Mi&r@+@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice�t���?!�ghǜn)@)�t���?1�ghǜn)@:Preprocessing2U
Iterator::Model::ParallelMapV2Œr�9>�?!g����(@)Œr�9>�?1g����(@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��DR��?!�>��F�R@)�:pΈ�~?1��P8�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor#-��#�v?!(�Khx?@)#-��#�v?1(�Khx?@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 52.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�32.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�O7G��U@Q߀E���+@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�<,Ԛ�?�<,Ԛ�?!�<,Ԛ�?      ��!       "	�*O ��?�*O ��?!�*O ��?*      ��!       2	鷯�?鷯�?!鷯�?:	4ڪ$��?4ڪ$��?!4ڪ$��?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�O7G��U@y߀E���+@