#pragma once

#include <torch/extension.h>
#include <ATen/div_rtn.h>
#include <ATen/native/utils/ParamUtils.h>

void check_deform_conv_backend(
	const at::Tensor& input,
	const at::Tensor& weight,
	const at::Tensor& offset_field,
	const at::Tensor& attn_mask,
	const at::Tensor& bias,
	const at::Tensor& grad_output,
	at::Backend location);

template<int64_t dim>
void check_deform_conv_shape(
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
	const at::Tensor& bias
);

template <int64_t dim>
std::vector<int64_t> get_output_size(
	const at::Tensor& input,
	at::IntArrayRef kernel_size,
	at::IntArrayRef stride_size,
	at::IntArrayRef pad_size,
	at::IntArrayRef dilation_size);

template <int64_t dim>
std::vector<int64_t> get_output_size(
	const at::Tensor& input,
	const at::Tensor& weight,
	at::IntArrayRef kernel_size,
	at::IntArrayRef stride_size,
	at::IntArrayRef pad_size,
	at::IntArrayRef dilation_size);

template <int64_t dim>
std::vector<int64_t> get_output_size(
	const at::Tensor& input,
	at::IntArrayRef kernel_size,
	at::IntArrayRef stride_size,
	at::IntArrayRef pad_size,
	at::IntArrayRef dilation_size) {
	std::vector<int64_t> sizes;
	for (const auto index : c10::irange(dim)) {
		sizes.push_back(
			div_rtn<int64_t>(
				input.size(index + input.dim() - dim) + 2 * pad_size[index] -
				(dilation_size[index] * (kernel_size[index] - 1) + 1),
				stride_size[index]) +
			1);
	}
	return sizes;
}

template <int64_t dim>
std::vector<int64_t> get_output_size(
	const at::Tensor& input,
	const at::Tensor& weight,
	at::IntArrayRef kernel_size,
	at::IntArrayRef stride_size,
	at::IntArrayRef pad_size,
	at::IntArrayRef dilation_size) {
	auto output_size = get_output_size<dim>(
		input, kernel_size, stride_size, pad_size, dilation_size);
	output_size.insert(output_size.begin(), weight.size(0));
	if (input.dim() == dim + 2) {
		output_size.insert(output_size.begin(), input.size(0));
	}
	return output_size;
}


template<int64_t dim>
void check_deform_conv_shape(
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
	const at::Tensor& bias
	) {

	// check params shape
	// input.shape == [batch, in_ch, *input_size]
	// weight.shape == [out_ch, in_ch / groups, *kernel_size]
	// offset_field.shape == [batch, dim, in_ch, *kernel_size, *input_size]
	// attn_amsk.shape == [batch, 1, in_ch, *kernel_size, *input_size]
	// bias.shape == [out_ch]

	// example, dim = 2,
	// input.shape == [batch, in_ch, h, w]
	// weight.shape == [out_ch, in_ch / groups, kH, kW]
	// offset_field.shape == [batch, dim, in_ch * kH * kW, h, w]
	// attn_amsk.shape == [batch, dim, in_ch * kH * kW, h, w]
	// bias.shape == [out_ch]

	TORCH_CHECK(input.dim() == 2 + dim);	
	TORCH_CHECK(weight.dim() == 2 + dim);
	TORCH_CHECK(offset_field.dim() == 3 + dim);
	TORCH_CHECK(attn_mask.dim() == 3 + dim);

	TORCH_CHECK(kernel_size.size() == dim);
	TORCH_CHECK(stride.size() == dim);
	TORCH_CHECK(padding.size() == dim);
	TORCH_CHECK(dilation.size() == dim);

	auto input_shape = input.sizes();
	auto output_shape = get_output_size<dim>(input, weight, kernel_size, stride, padding, dilation);
	auto weight_shape = weight.sizes();
	auto offset_field_shape = offset_field.sizes();
	auto attn_mask_shape = attn_mask.sizes();

	for (int64_t i = 0; i < dim; i++)
	{
		TORCH_CHECK(output_shape[2 + i] == offset_field_shape[3 + i]);
		TORCH_CHECK(output_shape[2 + i] == attn_mask_shape[3 + i]);
		TORCH_CHECK(weight_shape[2 + i] == kernel_size[i]);
	}

	int64_t kernel_sizes = c10::multiply_integers(kernel_size);

	TORCH_CHECK(input_shape[1] % groups == 0);
	TORCH_CHECK(output_shape[1] % groups == 0);
	TORCH_CHECK(weight_shape[1] == input_shape[1] / groups);
	TORCH_CHECK(offset_field_shape[2] == kernel_sizes * weight_shape[1] * groups);
	TORCH_CHECK(attn_mask_shape[2] == kernel_sizes * weight_shape[1] * groups);
	TORCH_CHECK(offset_field_shape[1] == dim);
	TORCH_CHECK(attn_mask_shape[1] == 1);
	

	if (bias.defined())
	{
		auto bias_shape = bias.sizes();
		TORCH_CHECK(bias.dim() == 1);
		TORCH_CHECK(bias_shape[0] == output_shape[1]);
	}

	if (grad_output.defined())
	{
		TORCH_CHECK(output_shape == grad_output.sizes());
	}
}