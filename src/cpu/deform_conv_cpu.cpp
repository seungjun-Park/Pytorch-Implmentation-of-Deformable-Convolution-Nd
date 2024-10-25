#include <torch/extension.h>

#include <cpu/im2col_cpu.h>
#include <cpu/col2im_cpu.h>

#include <deform_conv_utils.h>
#include <array_utils.h>
#include <type_utils.h>

template<int8_t dim>
at::Tensor deform_conv_nd_forward_cpu(
	const at::Tensor& input,
	const at::Tensor& weight,
	const at::Tensor& offset_field,
	const at::Tensor& attn_mask,
	at::IntArrayRef kernel_size,
	at::IntArrayRef stride,
	at::IntArrayRef padding,
	at::IntArrayRef dilation,
	const int64_t groups,
	const int64_t offset_field_channels_per_groups,
	const at::Tensor& bias) 
{
	auto k = weight.dim();
	int8_t tensor_dim = k - 2;

	TORCH_CHECK(dim == tensor_dim);

	at::Tensor undefined;

	check_deform_conv_backend(
		input,
		weight,
		offset_field,
		attn_mask,
		bias,
		undefined,
		at::Backend::CPU
	);

	at::Tensor output = at::zeros(get_output_size<dim>(input, weight, kernel_size, stride, padding, dilation), input.options().memory_format(at::MemoryFormat::Contiguous));

	// slice tensor sizes (b, c, *) to (*) 
	auto input_size = input.sizes();
	auto output_size = output.sizes();

	int32_t batch_size = input.size(0);
	int32_t in_channels = input.size(1);
	int32_t out_channels = weight.size(0);
	int32_t grouped_in_channels = in_channels / groups;
	int32_t grouped_out_channels = out_channels / groups;

	int32_t kernel_sizes = c10::multiply_integers(kernel_size);
	int32_t output_sizes = c10::multiply_integers(output_size.slice(2));

	at::Tensor columns = at::zeros({ groups, kernel_sizes * grouped_in_channels, output_sizes }, input.options().memory_format(at::MemoryFormat::Contiguous));

	AT_DISPATCH_FLOATING_TYPES_AND(at::kBFloat16, input.scalar_type(), "deform_conv_nd_forward<>", [&]() {

		using scalar_t = scalar_t;

		for (const auto b : c10::irange(batch_size))
		{
			at::Tensor input_n = input.select(0, b);
			at::Tensor offset_field_n = offset_field.select(0, b);
			at::Tensor attn_mask_n = attn_mask.select(0, b);

			im2col_nd_cpu<scalar_t, dim>(
				input_n.const_data_ptr<scalar_t>(),
				offset_field_n.const_data_ptr<scalar_t>(),
				attn_mask_n.const_data_ptr<scalar_t>(),
				grouped_in_channels,
				IntArrayRef2IntArray<dim>(input_size.slice(2)),
				IntArrayRef2IntArray<dim>(output_size.slice(2)),
				IntArrayRef2IntArray<dim>(kernel_size),
				IntArrayRef2IntArray<dim>(stride),
				IntArrayRef2IntArray<dim>(padding),
				IntArrayRef2IntArray<dim>(dilation),
				groups,
				offset_field_channels_per_groups,
				columns.mutable_data_ptr<scalar_t>()
			);

			// torch::bmm was not implemented for fp16 in cpu, so we convert fp16 to fp32
			output.select(0, b) = torch::bmm(
				weight.reshape({ groups, grouped_out_channels, -1 }),
				columns
			).reshape(output_size.slice(1));
		}

		// add bias
		if (bias.defined())
		{
			output = (output.reshape({ batch_size, out_channels, -1 }).transpose(1, 2) + bias).transpose(1, 2).reshape(output_size);
		}
		});

	return output;
}

template<int8_t dim>
torch::autograd::tensor_list deform_conv_nd_backward_cpu(
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
	const int64_t offset_field_channels_per_groups,
	const at::Tensor& bias) {

	auto k = weight.dim();
	int8_t tensor_dim = k - 2;

	TORCH_CHECK(dim == tensor_dim);

	check_deform_conv_backend(
		input,
		weight,
		offset_field,
		attn_mask,
		bias,
		grad_output,
		at::Backend::CPU
	);

	at::Tensor output = at::zeros(get_output_size<dim>(input, weight, kernel_size, stride, padding, dilation), input.options().memory_format(at::MemoryFormat::Contiguous));

	at::Tensor grad_input = at::zeros_like(input);
	at::Tensor grad_weight = at::zeros_like(weight);
	at::Tensor grad_offset_field = at::zeros_like(offset_field);
	at::Tensor grad_attn_mask = at::zeros_like(attn_mask);
	at::Tensor grad_bias = bias.defined() ? at::zeros_like(bias) : at::Tensor();

	auto input_size = input.sizes();
	auto output_size = output.sizes();

	int32_t batch_size = input.size(0);
	int32_t in_channels = input.size(1);
	int32_t out_channels = weight.size(0);
	int32_t grouped_in_channels = in_channels / groups;
	int32_t grouped_out_channels = out_channels / groups;

	int32_t kernel_sizes = c10::multiply_integers(kernel_size);
	int32_t output_sizes = c10::multiply_integers(output_size.slice(2));

	AT_DISPATCH_FLOATING_TYPES_AND(at::kBFloat16, input.scalar_type(), "deform_conv_nd_backward<>", [&]() {

		using scalar_t = scalar_t;

		for (const auto b : c10::irange(batch_size))
		{
			at::Tensor input_n = input.select(0, b);
			at::Tensor offset_field_n = offset_field.select(0, b);
			at::Tensor attn_mask_n = attn_mask.select(0, b);

			at::Tensor grad_input_n = grad_input.select(0, b);
			at::Tensor grad_offset_field_n = grad_offset_field.select(0, b);
			at::Tensor grad_attn_mask_n = grad_attn_mask.select(0, b);
			at::Tensor grad_output_n = grad_output.select(0, b);

			// compute col = weight^T * grad_output 
			// torch::bmm was not implemented for fp16 in cpu, so we convert fp16 to fp32
			at::Tensor columns = torch::bmm(
				weight.reshape({ groups, grouped_out_channels, -1 }).transpose(1, 2),
				grad_output_n.reshape({ groups, grouped_out_channels, -1 })
			);

			// compute gradient of inputs, offset_field, attn_mask
			col2im_nd_cpu<scalar_t, dim>(
				input_n.const_data_ptr<scalar_t>(),
				columns.const_data_ptr<scalar_t>(),
				offset_field_n.const_data_ptr<scalar_t>(),
				attn_mask_n.const_data_ptr<scalar_t>(),
				grouped_in_channels,
				IntArrayRef2IntArray<dim>(input_size.slice(2)),
				IntArrayRef2IntArray<dim>(output_size.slice(2)),
				IntArrayRef2IntArray<dim>(kernel_size),
				IntArrayRef2IntArray<dim>(stride),
				IntArrayRef2IntArray<dim>(padding),
				IntArrayRef2IntArray<dim>(dilation),
				groups,
				offset_field_channels_per_groups,
				(mapped_type<scalar_t>*)grad_input_n.mutable_data_ptr<scalar_t>(),
				(mapped_type<scalar_t>*)grad_offset_field_n.mutable_data_ptr<scalar_t>(),
				(mapped_type<scalar_t>*)grad_attn_mask_n.mutable_data_ptr<scalar_t>()
			);

			// compute gradient of weight.

			im2col_nd_cpu<scalar_t, dim>(
				input_n.const_data_ptr<scalar_t>(),
				offset_field_n.const_data_ptr<scalar_t>(),
				attn_mask_n.const_data_ptr<scalar_t>(),
				grouped_in_channels,
				IntArrayRef2IntArray<dim>(input_size.slice(2)),
				IntArrayRef2IntArray<dim>(output_size.slice(2)),
				IntArrayRef2IntArray<dim>(kernel_size),
				IntArrayRef2IntArray<dim>(stride),
				IntArrayRef2IntArray<dim>(padding),
				IntArrayRef2IntArray<dim>(dilation),
				groups,
				offset_field_channels_per_groups,
				columns.mutable_data_ptr<scalar_t>()
			);

			// compute grad_out grad_output * col^T
			// torch::bmm was not implemented for fp16 in cpu, so we convert fp16 to fp32
			grad_weight += torch::bmm(
				grad_output_n.reshape({ groups, grouped_out_channels, -1 }),
				columns.transpose(1, 2)
			).reshape(grad_weight.sizes());
		}

		// compute gradient of bias(if defined)

		if (grad_bias.defined())
		{
			std::vector<int64_t> dims(dim + 1, 0);
			std::iota(dims.begin() + 1, dims.end(), 2);
			grad_bias = grad_output.sum(dims);
		}

		});

	return { grad_input, grad_weight, grad_offset_field, grad_attn_mask, grad_bias };
}

TORCH_LIBRARY_IMPL(custom_op, CPU, m)
{
	m.impl("deform_conv1d_forward", &deform_conv_nd_forward_cpu<1>);
	m.impl("deform_conv2d_forward", &deform_conv_nd_forward_cpu<2>);
	m.impl("deform_conv3d_forward", &deform_conv_nd_forward_cpu<3>);
	
	m.impl("deform_conv1d_backward", &deform_conv_nd_backward_cpu<1>);
	m.impl("deform_conv2d_backward", &deform_conv_nd_backward_cpu<2>);
	m.impl("deform_conv3d_backward", &deform_conv_nd_backward_cpu<3>);
}