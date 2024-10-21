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
