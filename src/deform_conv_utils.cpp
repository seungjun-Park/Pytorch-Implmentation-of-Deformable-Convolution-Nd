#include <deform_conv_utils.h>

void check_deform_conv_backend(
	const at::Tensor& input,
	const at::Tensor& weight,
	const at::Tensor& offset_field,
	const at::Tensor& attn_mask,
	const at::Tensor& bias,
	const at::Tensor& grad_output,
	at::Backend location)
{

	at::checkBackend("check_deform_conv_backend", { input, weight, offset_field, attn_mask }, location);

	if (bias.defined())
	{
		at::checkBackend("check_deform_conv_backend", { bias }, location);
	}
	if (grad_output.defined())
	{
		at::checkBackend("check_deform_conv_backend", { grad_output }, location);
	}
}