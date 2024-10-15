#pragma once
#include <torch/extension.h>

#include <vector>

at::Tensor deform_conv_forward(
	const at::Tensor& input,
	const at::Tensor& weight,
	const at::Tensor& offset_field,
	const at::Tensor& attn_mask,
	at::IntArrayRef kernel_size,
	at::IntArrayRef stride = 1,
	at::IntArrayRef padding = 0,
	at::IntArrayRef dilation = 1,
	int64_t groups = 1,
	const c10::optional<at::Tensor>& bias_opt = {}
);

std::vector<at::Tensor> deform_conv_backward(
	const at::Tensor& input,
	const at::Tensor& weight,
	const at::Tensor& offset_field,
	const at::Tensor& attn_mask,
	const at::Tensor& grad_output,
	at::IntArrayRef kernel_size,
	at::IntArrayRef stride = 1,
	at::IntArrayRef padding = 0,
	at::IntArrayRef dilation = 1,
	int64_t groups = 1,
	const c10::optional<at::Tensor>& bias_opt = {}
);