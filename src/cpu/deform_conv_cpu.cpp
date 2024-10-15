#pragma once
#include <cpu/deform_conv_cpu.h>
#include <cpu/im2col_cpu.h>
#include <cpu/col2im_cpu.h>

#include <deform_conv_utils.h>
#include <utils.h>

template<int64_t dim>
at::Tensor _deform_conv_nd_forward_cpu(
	const at::Tensor& input,
	const at::Tensor& weight,
	const at::Tensor& offset_field,
	const at::Tensor& attn_mask,
	at::IntArrayRef kernel_size,
	at::IntArrayRef stride,
	at::IntArrayRef padding,
	at::IntArrayRef dilation,
	int64_t groups,
	const at::Tensor& bias) {

	at::Tensor output = at::zeros(get_output_size<dim>(input, weight, kernel_size, stride, padding, dilation), input.options().memory_format(at::MemoryFormat::Contiguous));

	at::Tensor undefined;

	/*check_deform_conv_shape<dim>(
		input,
		weight,
		offset_field,
		attn_mask,
		undefined,
		kernel_size,
		stride,
		padding,
		dilation,
		groups,
		bias
	);*/

	// slice tensor sizes (b, c, *) to (*) 
	auto input_size = input.sizes();
	auto output_size = output.sizes();

	int64_t batch_size = input.size(0);
	int64_t in_channels = input.size(1);
	int64_t out_channels = weight.size(0);
	int64_t grouped_in_channels = in_channels / groups;
	int64_t grouped_out_channels = out_channels / groups;

	int64_t kernel_sizes = c10::multiply_integers(kernel_size);
	int64_t output_sizes = c10::multiply_integers(output_size.slice(2));

	at::Tensor columns = at::empty({ groups, kernel_sizes * grouped_in_channels, output_sizes }, input.options().memory_format(at::MemoryFormat::Contiguous));

	AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "deform_conv_nd_forward<>", [&]() {

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
				columns.mutable_data_ptr<scalar_t>()
			);

			// output_n.shape = [groups, grouped_out_channels, output_sizes]
			output.select(0, b) = torch::bmm(weight.reshape({ groups, grouped_out_channels, -1 }), columns).reshape(output_size.slice(1));
		}

		// add bias
		if (bias.defined())
		{
			output = output.reshape({ batch_size, out_channels, -1 }).transpose(1, 2);
			output += bias;
			output = output.transpose(1, 2).reshape(output_size);
		}
		});

	return output;
}

at::Tensor deform_conv_nd_forward_cpu(
	const at::Tensor& input,
	const at::Tensor& weight,
	const at::Tensor& offset_field,
	const at::Tensor& attn_mask,
	at::IntArrayRef kernel_size,
	at::IntArrayRef stride,
	at::IntArrayRef padding,
	at::IntArrayRef dilation,
	int64_t groups,
	const at::Tensor& bias)
{
	TORCH_CHECK(input.is_cpu());

	auto k = weight.dim();
	int64_t dim = k - 2;

	// the function pointer for each dimension implementation
	at::Tensor(*deform_conv_nd_forward_cpu_func)(
		const at::Tensor & input,
		const at::Tensor & weight,
		const at::Tensor & offset_field,
		const at::Tensor & attn_mask,
		at::IntArrayRef kernel_size,
		at::IntArrayRef stride,
		at::IntArrayRef padding,
		at::IntArrayRef dilation,
		int64_t groups,
		const at::Tensor & bias) = nullptr;

	switch (dim)
	{
	case 1:
		deform_conv_nd_forward_cpu_func = _deform_conv_nd_forward_cpu<1>;
		break;
	case 2:
		deform_conv_nd_forward_cpu_func = _deform_conv_nd_forward_cpu<2>;
		break;
	case 3:
		deform_conv_nd_forward_cpu_func = _deform_conv_nd_forward_cpu<3>;
		break;
	}

	at::Tensor output = (*deform_conv_nd_forward_cpu_func)(
		input,
		weight,
		offset_field,
		attn_mask,
		kernel_size,
		stride,
		padding,
		dilation,
		groups,
		bias
		);

	return output;
}

template<int64_t dim>
std::vector<at::Tensor> _deform_conv_nd_backward_cpu(
	const at::Tensor& input,
	const at::Tensor& weight,
	const at::Tensor& offset_field,
	const at::Tensor& attn_mask,
	const at::Tensor& grad_output,
	at::IntArrayRef kernel_size,
	at::IntArrayRef stride,
	at::IntArrayRef padding,
	at::IntArrayRef dilation,
	int64_t groups,
	const at::Tensor& bias) {

	/*check_deform_conv_shape<dim>(
		input,
		weight,
		offset_field,
		attn_mask,
		grad_output,
		kernel_size,
		stride,
		padding,
		dilation,
		groups,
		bias
	);*/

	at::Tensor output = at::zeros(get_output_size<dim>(input, weight, kernel_size, stride, padding, dilation), input.options().memory_format(at::MemoryFormat::Contiguous));

	at::Tensor grad_input = at::zeros_like(input);
	at::Tensor grad_weight = at::zeros_like(weight);
	at::Tensor grad_offset_field = at::zeros_like(offset_field);
	at::Tensor grad_attn_mask = at::zeros_like(attn_mask);
	at::Tensor grad_bias = bias.defined() ? at::zeros_like(bias) : at::Tensor();

	auto input_size = input.sizes();
	auto output_size = output.sizes();

	int64_t batch_size = input.size(0);
	int64_t in_channels = input.size(1);
	int64_t out_channels = weight.size(0);
	int64_t grouped_in_channels = in_channels / groups;
	int64_t grouped_out_channels = out_channels / groups;

	int64_t kernel_sizes = c10::multiply_integers(kernel_size);
	int64_t output_sizes = c10::multiply_integers(output_size.slice(2));

	at::Tensor columns = at::empty({ groups, kernel_sizes * grouped_in_channels, output_sizes }, input.options().memory_format(at::MemoryFormat::Contiguous));

	AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "deform_conv_nd_backward<>", [&]() {

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
			columns = torch::bmm(weight.reshape({ groups, grouped_out_channels, -1 }).transpose(1, 2), grad_output_n.reshape({ groups, grouped_out_channels, -1 }));

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
				grad_input_n.mutable_data_ptr<scalar_t>(),
				grad_offset_field_n.mutable_data_ptr<scalar_t>(),
				grad_attn_mask_n.mutable_data_ptr<scalar_t>()
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
				columns.mutable_data_ptr<scalar_t>()
			);

			// compute grad_out grad_output * col^T
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

	return std::vector<at::Tensor>({ grad_input, grad_weight, grad_offset_field, grad_attn_mask, grad_bias });
}

std::vector<at::Tensor> deform_conv_nd_backward_cpu(
	const at::Tensor& input,
	const at::Tensor& weight,
	const at::Tensor& offset_field,
	const at::Tensor& attn_mask,
	const at::Tensor& grad_output,
	at::IntArrayRef kernel_size,
	at::IntArrayRef stride,
	at::IntArrayRef padding,
	at::IntArrayRef dilation,
	int64_t groups,
	const at::Tensor& bias)
{
	TORCH_CHECK(input.is_cpu());

	auto k = weight.dim();
	int64_t dim = k - 2;

	// the function pointer for each dimension implementation
	std::vector<at::Tensor>(*deform_conv_nd_backward_cpu_func)(
		const at::Tensor & input,
		const at::Tensor & weight,
		const at::Tensor & offset_field,
		const at::Tensor & attn_mask,
		const at::Tensor & grad_output,
		at::IntArrayRef kernel_size,
		at::IntArrayRef stride,
		at::IntArrayRef padding,
		at::IntArrayRef dilation,
		int64_t groups,
		const at::Tensor & bias) = nullptr;

	switch (dim)
	{
	case 1:
		deform_conv_nd_backward_cpu_func = _deform_conv_nd_backward_cpu<1>;
		break;
	case 2:
		deform_conv_nd_backward_cpu_func = _deform_conv_nd_backward_cpu<2>;
		break;
	case 3:
		deform_conv_nd_backward_cpu_func = _deform_conv_nd_backward_cpu<3>;
		break;
	}

	std::vector<at::Tensor> grads = (*deform_conv_nd_backward_cpu_func)(
		input,
		weight,
		offset_field,
		attn_mask,
		grad_output,
		kernel_size,
		stride,
		padding,
		dilation,
		groups,
		bias
		);

	return grads;
}