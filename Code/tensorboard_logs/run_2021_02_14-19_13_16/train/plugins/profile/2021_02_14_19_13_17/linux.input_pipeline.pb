	��.R(�#@��.R(�#@!��.R(�#@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-��.R(�#@���E�@1�Ac&Q�?A�Ά�3��?I�g���?*		�Zd[Z@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatu��OU�?!��3�5@@)e��2�P�?1帔p5::@:Preprocessing2F
Iterator::Modelޭ,�Yf�?!�1v;�@@)�R����?1@�8�&'1@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapa��L��?!��O�_=@)��iO�9�?15�om�/@:Preprocessing2U
Iterator::Model::ParallelMapV2���H�?!�g��).@)���H�?1�g��).@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice�'�����?!KJC0��*@)�'�����?1KJC0��*@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�~k'J�?!�D� �P@)m�{?1�&�3�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor0�1"Qhy?!�CK�׈@)0�1"Qhy?1�CK�׈@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 80.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�15.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noID��F��W@Q�+�[�@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���E�@���E�@!���E�@      ��!       "	�Ac&Q�?�Ac&Q�?!�Ac&Q�?*      ��!       2	�Ά�3��?�Ά�3��?!�Ά�3��?:	�g���?�g���?!�g���?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qD��F��W@y�+�[�@