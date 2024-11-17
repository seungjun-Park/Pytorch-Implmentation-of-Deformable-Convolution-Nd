#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda/im2col_cuda.h>
#include <cuda/col2im_cuda.h>
#include <cuda_runtime.h>

#include <GPUInfo.h>
#include <deform_conv_utils.h>
#include <array_utils.h>
#include <type_utils.h>

template<int8_t dim>
at::Tensor deform_conv_nd_forward_cuda(
	const at::Tensor& input,
	const at::Tensor& weight,
	const at::Tensor& offset_field,
	const at::Tensor& attn_mask,
	at::IntArrayRef kernel_size,
	at::IntArrayRef stride,
	at::IntArrayRef padding,
	at::IntArrayRef dilation,
	const int64_t groups,
	const int64_t deformable_groups_per_groups, 
	const double_t offset_scale,
	const bool fix_center,
	const at::Tensor& bias) {

	auto k = weight.dim();
	int64_t tensor_dim = k - 2;

	TORCH_CHECK(dim == tensor_dim);

	at::Tensor undefined;

	check_deform_conv_backend(
		input,
		weight,
		offset_field,
		attn_mask,
		bias,
		undefined,
		at::Backend::CUDA
	);

	bool is_channels_last = check_is_channels_last<dim>(input);

	// slice tensor sizes (b, c, *) to (*) 
	auto input_size = input.sizes();

	int64_t batch_size = input.size(0);
	int64_t in_channels = input.size(1);
	int64_t out_channels = weight.size(0);
	int64_t grouped_in_channels = in_channels / groups;
	int64_t grouped_out_channels = out_channels / groups;

	at::Tensor output = at::zeros(
		get_output_size<dim>(input, weight, kernel_size, stride, padding, dilation),
		input.options()
	).contiguous();

	if (is_channels_last)
	{
		auto output_size = output.sizes();
		output = output.reshape({ batch_size, out_channels, -1 }).transpose(1, 2).contiguous().transpose(1, 2).reshape(output_size);
	}

	auto output_size = output.sizes();

	int64_t kernel_sizes = c10::multiply_integers(kernel_size);
	int64_t output_sizes = c10::multiply_integers(output_size.slice(2));

	if (fix_center)
	{
		// fix_center only available when kernel sizes is odd.
		TORCH_CHECK(kernel_sizes % 2 != 0);
	}

	torch::Device device = input.device();

	TORCH_CHECK(device.index() < torch::cuda::device_count());

	// custom class to check current gpu status.

	GPUInfo gpu_info;
	auto device_properties = gpu_info.GetDeviceProps()[device.index()];

	int64_t columns_numel = groups * batch_size * output_sizes * grouped_in_channels * kernel_sizes;
	int64_t per_elements_in_batch = groups * output_sizes * grouped_in_channels * kernel_sizes;

	int32_t min_grid_size, block_size;
	AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, input.scalar_type(), "get_blocks", [&]() {
		using scalar_t = scalar_t;

		auto im2col_nd_cuda_func = im2col_nd_cuda<scalar_t, dim, false>;

		if (is_channels_last)
		{
			im2col_nd_cuda_func = im2col_nd_cuda<scalar_t, dim, true>;
		}

		cudaOccupancyMaxPotentialBlockSize(
			&min_grid_size,
			&block_size,
			im2col_nd_cuda_func,
			0,
			device_properties.maxThreadsPerBlock);
	});

	int32_t num_blocks = (columns_numel + block_size - 1) / block_size;

	int32_t sub_batch_size = (num_blocks * block_size) / per_elements_in_batch;
	int32_t total_iteration = batch_size / sub_batch_size;

	at::Tensor columns = at::zeros({ groups * grouped_in_channels * kernel_sizes, sub_batch_size * output_sizes }, input.options()).contiguous();

	if (is_channels_last)
	{
		columns = columns.transpose(0, 1).contiguous().transpose(0, 1);
	}

	columns = columns.reshape({ groups, -1, sub_batch_size * output_sizes });

	std::vector<int64_t> output_n_size(2 + dim);
	output_n_size[0] = out_channels;
	output_n_size[1] = sub_batch_size;
	for (int8_t i = 0; i < dim; i++)
	{
		output_n_size[2 + i] = output_size.slice(2)[i];
	}

	auto cudaStream = at::cuda::getCurrentCUDAStream(device.index());

	AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, input.scalar_type(), "deform_conv_nd_forward<>", [&]() {
		using scalar_t = scalar_t;
		auto im2col_nd_cuda_func = im2col_nd_cuda<scalar_t, dim, false>;

		if (is_channels_last)
		{
			im2col_nd_cuda_func = im2col_nd_cuda<scalar_t, dim, true>;
		}

		for (const auto n : c10::irange(total_iteration))
		{
			columns.zero_();

			int64_t batch_start = sub_batch_size * n;
			at::Tensor input_n = input.slice(0, batch_start, batch_start + sub_batch_size);
			at::Tensor offset_field_n = offset_field.slice(0, batch_start, batch_start + sub_batch_size);
			at::Tensor attn_mask_n = attn_mask.slice(0, batch_start, batch_start + sub_batch_size);

			im2col_nd_cuda_func<<<num_blocks, block_size, 0, cudaStream >>>(
				input_n.const_data_ptr<scalar_t>(),
				offset_field_n.const_data_ptr<scalar_t>(),
				attn_mask_n.const_data_ptr<scalar_t>(),
				sub_batch_size,
				grouped_in_channels,
				IntArrayRef2IntArray<dim>(input_size.slice(2)),
				IntArrayRef2IntArray<dim>(output_size.slice(2)),
				IntArrayRef2IntArray<dim>(kernel_size),
				IntArrayRef2IntArray<dim>(stride),
				IntArrayRef2IntArray<dim>(padding),
				IntArrayRef2IntArray<dim>(dilation),
				groups,
				deformable_groups_per_groups,
				offset_scale,
				fix_center,
				columns.mutable_data_ptr<scalar_t>()
			);

			output.slice(0, batch_start, batch_start + sub_batch_size).copy_(torch::bmm(
				weight.reshape({ groups, grouped_out_channels, -1 }),
				columns
			).reshape(output_n_size).transpose(0, 1));
		}
		
		if (bias.defined())
		{
			output = (output.reshape({ batch_size, out_channels, -1 }).transpose(1, 2) + bias).transpose(1, 2).reshape(output_size);
		}
		});

	return output;
}

template<int8_t dim>
torch::autograd::tensor_list deform_conv_nd_backward_cuda(
	const at::Tensor& input,
	const at::Tensor& weight,
	const at::Tensor& offset_field,
	const at::Tensor& attn_mask,
	const at::Tensor& grad_output,
	at::IntArrayRef kernel_size,
	at::IntArrayRef stride,
	at::IntArrayRef padding,
	at::IntArrayRef dilation,
	const int64_t groups,
	const int64_t deformable_groups_per_groups,
	const double_t offset_scale,
	const bool fix_center,
	const at::Tensor& bias) {

	auto k = weight.dim();
	int64_t tensor_dim = k - 2;

	TORCH_CHECK(dim == tensor_dim);

	check_deform_conv_backend(
		input,
		weight,
		offset_field,
		attn_mask,
		bias,
		grad_output,
		at::Backend::CUDA
	);

	bool is_channels_last = check_is_channels_last<dim>(input);

	at::Tensor grad_input = at::zeros_like(input);
	at::Tensor grad_weight = at::zeros_like(weight);
	at::Tensor grad_offset_field = at::zeros_like(offset_field);
	at::Tensor grad_attn_mask = at::zeros_like(attn_mask);
	at::Tensor grad_bias = bias.defined() ? at::zeros_like(bias) : at::Tensor();

	// slice tensor sizes (b, c, *) to (*) 
	auto input_size = input.sizes();

	int64_t batch_size = input.size(0);
	int64_t in_channels = input.size(1);
	int64_t out_channels = weight.size(0);
	int64_t grouped_in_channels = in_channels / groups;
	int64_t grouped_out_channels = out_channels / groups;

	at::Tensor output = at::zeros(
		get_output_size<dim>(input, weight, kernel_size, stride, padding, dilation),
		input.options()
	).contiguous();

	if (is_channels_last)
	{
		auto output_size = output.sizes();
		output = output.reshape({ batch_size, out_channels, -1 }).transpose(1, 2).contiguous().transpose(1, 2).reshape(output_size);
	}

	auto output_size = output.sizes();

	int64_t kernel_sizes = c10::multiply_integers(kernel_size);
	int64_t output_sizes = c10::multiply_integers(output_size.slice(2));

	if (fix_center)
	{
		// fix_center only available when kernel sizes is odd.
		TORCH_CHECK(kernel_sizes % 2 != 0);
	}

	torch::Device device = input.device();

	TORCH_CHECK(device.index() < torch::cuda::device_count());

	GPUInfo gpu_info;
	auto device_properties = gpu_info.GetDeviceProps()[device.index()];

	int64_t columns_numel = groups * batch_size * output_sizes * grouped_in_channels * kernel_sizes;
	int64_t per_elements_in_batch = groups * output_sizes * grouped_in_channels * kernel_sizes;

	int32_t min_grid_size, block_size;
	AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, input.scalar_type(), "get_blocks", [&]() {
		using scalar_t = scalar_t;
		int32_t min_grid_size_im2col, block_size_im2col;
		int32_t min_grid_size_col2im, block_size_col2im;
		auto im2col_nd_cuda_func = im2col_nd_cuda<scalar_t, dim, false>;
		auto col2im_nd_cuda_func = col2im_nd_cuda<scalar_t, dim, false>;
		if (is_channels_last)
		{
			im2col_nd_cuda_func = im2col_nd_cuda<scalar_t, dim, true>;
			col2im_nd_cuda_func = col2im_nd_cuda<scalar_t, dim, true>;
		}
		
		cudaOccupancyMaxPotentialBlockSize(
			&min_grid_size_im2col, 
			&block_size_im2col, 
			im2col_nd_cuda_func, 
			0,
			device_properties.maxThreadsPerBlock);

		cudaOccupancyMaxPotentialBlockSize(
			&min_grid_size_col2im, 
			&block_size_col2im, 
			col2im_nd_cuda_func, 
			0,
			device_properties.maxThreadsPerBlock);

		if (block_size_col2im < block_size_im2col)
		{
			min_grid_size = min_grid_size_col2im;
			block_size = block_size_col2im;
		}
		else
		{
			min_grid_size = min_grid_size_im2col;
			block_size = block_size_im2col;
		}
		});

	int32_t num_blocks = (columns_numel + block_size - 1) / block_size;

	int64_t sub_batch_size = (num_blocks * block_size) / per_elements_in_batch;
	int64_t total_iteration = batch_size / sub_batch_size;

	at::Tensor columns = at::zeros({ groups * grouped_in_channels * kernel_sizes, sub_batch_size * output_sizes }, input.options()).contiguous();

	if (is_channels_last)
	{
		columns = columns.transpose(0, 1).contiguous().transpose(0, 1);
	}

	columns = columns.reshape({ groups, -1, sub_batch_size * output_sizes });

	auto cudaStream = at::cuda::getCurrentCUDAStream(device.index());

	AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, input.scalar_type(), "deform_conv_nd_backward<>", [&]() {
		using scalar_t = scalar_t;

		auto im2col_nd_cuda_func = im2col_nd_cuda<scalar_t, dim, false>;
		auto col2im_nd_cuda_func = col2im_nd_cuda<scalar_t, dim, false>;
		if (is_channels_last)
		{
			im2col_nd_cuda_func = im2col_nd_cuda<scalar_t, dim, true>;
			col2im_nd_cuda_func = col2im_nd_cuda<scalar_t, dim, true>;
		}

		for (const auto n : c10::irange(total_iteration))
		{
			int64_t batch_start = sub_batch_size * n;

			at::Tensor input_n = input.slice(0, batch_start, batch_start + sub_batch_size);
			at::Tensor offset_field_n = offset_field.slice(0, batch_start, batch_start + sub_batch_size);
			at::Tensor attn_mask_n = attn_mask.slice(0, batch_start, batch_start + sub_batch_size);

			at::Tensor grad_input_n = grad_input.slice(0, batch_start, batch_start + sub_batch_size);
			at::Tensor grad_offset_field_n = grad_offset_field.slice(0, batch_start, batch_start + sub_batch_size);
			at::Tensor grad_attn_mask_n = grad_attn_mask.slice(0, batch_start, batch_start + sub_batch_size);
			at::Tensor grad_output_n = grad_output.slice(0, batch_start, batch_start + sub_batch_size);
			
			columns.zero_();

			columns.copy_(torch::bmm(
				weight.reshape({ groups, grouped_out_channels, -1 }).transpose(1, 2),
				grad_output_n.transpose(0, 1).reshape({ groups, grouped_out_channels, -1 })
			));

			// compute gradient of inputs, offset_field, attn_mask
			col2im_nd_cuda_func<<<num_blocks, block_size, 0, cudaStream >>> (
				input_n.const_data_ptr<scalar_t>(),
				columns.const_data_ptr<scalar_t>(),
				offset_field_n.const_data_ptr<scalar_t>(),
				attn_mask_n.const_data_ptr<scalar_t>(),
				sub_batch_size,
				grouped_in_channels,
				IntArrayRef2IntArray<dim>(input_size.slice(2)),
				IntArrayRef2IntArray<dim>(output_size.slice(2)),
				IntArrayRef2IntArray<dim>(kernel_size),
				IntArrayRef2IntArray<dim>(stride),
				IntArrayRef2IntArray<dim>(padding),
				IntArrayRef2IntArray<dim>(dilation),
				groups,
				deformable_groups_per_groups,
				offset_scale,
				fix_center,
				(mapped_type<scalar_t>*)grad_input_n.mutable_data_ptr<scalar_t>(),
				(mapped_type<scalar_t>*)grad_offset_field_n.mutable_data_ptr<scalar_t>(),
				(mapped_type<scalar_t>*)grad_attn_mask_n.mutable_data_ptr<scalar_t>()
				);

			columns.zero_();

			// compute grad_weight = grad_output * col^T
			im2col_nd_cuda_func<<<num_blocks, block_size, 0, cudaStream>>>(
				input_n.const_data_ptr<scalar_t>(),
				offset_field_n.const_data_ptr<scalar_t>(),
				attn_mask_n.const_data_ptr<scalar_t>(),
				sub_batch_size,
				grouped_in_channels,
				IntArrayRef2IntArray<dim>(input_size.slice(2)),
				IntArrayRef2IntArray<dim>(output_size.slice(2)),
				IntArrayRef2IntArray<dim>(kernel_size),
				IntArrayRef2IntArray<dim>(stride),
				IntArrayRef2IntArray<dim>(padding),
				IntArrayRef2IntArray<dim>(dilation),
				groups,
				deformable_groups_per_groups,
				offset_scale,
				fix_center,
				columns.mutable_data_ptr<scalar_t>()
			);

			// compute grad_weight = grad_output * col^T
			grad_weight += torch::bmm(
				grad_output_n.transpose(0, 1).reshape({groups, grouped_out_channels, -1}),
				columns.transpose(1, 2)
			).reshape(grad_weight.sizes());
		}

		if (grad_bias.defined())
		{
			std::vector<int64_t> dims(dim + 1, 0);
			std::iota(dims.begin() + 1, dims.end(), 2);
			grad_bias = grad_output.sum(dims);
		}
	});

	at::Tensor undefined;

	return {
		grad_input, grad_weight, grad_offset_field, grad_attn_mask,
		undefined, undefined, undefined, undefined, undefined, undefined, undefined, undefined,
		grad_bias
	};
}

TORCH_LIBRARY_IMPL(custom_op, CUDA, m)
{
	m.impl("deform_conv1d_forward", &deform_conv_nd_forward_cuda<1>);
	m.impl("deform_conv2d_forward", &deform_conv_nd_forward_cuda<2>);
	m.impl("deform_conv3d_forward", &deform_conv_nd_forward_cuda<3>);

	m.impl("deform_conv1d_backward", &deform_conv_nd_backward_cuda<1>);
	m.impl("deform_conv2d_backward", &deform_conv_nd_backward_cuda<2>);
	m.impl("deform_conv3d_backward", &deform_conv_nd_backward_cuda<3>);
}