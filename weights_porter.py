def port_weights(mb_model, torch_model):
    mb_model.prompt_encoder.background_point_embed.set_weights(
        [
            torch_model.prompt_encoder.point_embeddings[0]
            .weight.cpu()
            .detach()
            .numpy()
        ]
    )
    mb_model.prompt_encoder.foreground_point_embed.set_weights(
        [
            torch_model.prompt_encoder.point_embeddings[1]
            .weight.cpu()
            .detach()
            .numpy()
        ]
    )
    mb_model.prompt_encoder.top_left_corner_embed.set_weights(
        [
            torch_model.prompt_encoder.point_embeddings[2]
            .weight.cpu()
            .detach()
            .numpy()
        ]
    )
    mb_model.prompt_encoder.bottom_right_corner_embed.set_weights(
        [
            torch_model.prompt_encoder.point_embeddings[3]
            .weight.cpu()
            .detach()
            .numpy()
        ]
    )
    mb_model.prompt_encoder.not_a_point_embed.set_weights(
        [
            torch_model.prompt_encoder.not_a_point_embed.weight.cpu()
            .detach()
            .numpy()
        ]
    )
    mb_model.prompt_encoder.mask_downscaler.set_weights(
        [
            x.permute(2, 3, 1, 0).cpu().detach().numpy()
            if x.ndim == 4
            else x.cpu().detach().numpy()
            for x in torch_model.prompt_encoder.mask_downscaling.parameters()
        ]
    )
    mb_model.prompt_encoder.no_mask_embed.set_weights(
        [
            torch_model.prompt_encoder.no_mask_embed.weight.cpu()
            .detach()
            .numpy()
        ]
    )
    mb_model.prompt_encoder.positional_embedding_layer.positional_encoding_gaussian_matrix = (
        torch_model.prompt_encoder.pe_layer.positional_encoding_gaussian_matrix.cpu().numpy()
    )
    for i in range(2):
        mb_model.mask_decoder.transformer.layers[i].self_attention.set_weights(
            [
                x.cpu().detach().numpy().T
                for x in torch_model.mask_decoder.transformer.layers[
                    i
                ].self_attn.parameters()
            ]
        )
        mb_model.mask_decoder.transformer.layers[i].layer_norm1.set_weights(
            [
                x.cpu().detach().numpy()
                for x in torch_model.mask_decoder.transformer.layers[
                    i
                ].norm1.parameters()
            ]
        )
        mb_model.mask_decoder.transformer.layers[
            i
        ].cross_attention_token_to_image.set_weights(
            [
                x.cpu().detach().numpy().T
                for x in torch_model.mask_decoder.transformer.layers[
                    i
                ].cross_attn_token_to_image.parameters()
            ]
        )
        mb_model.mask_decoder.transformer.layers[i].layer_norm2.set_weights(
            [
                x.cpu().detach().numpy()
                for x in torch_model.mask_decoder.transformer.layers[
                    i
                ].norm2.parameters()
            ]
        )
        mb_model.mask_decoder.transformer.layers[i].mlp_block.set_weights(
            [
                x.cpu().detach().numpy().T
                for x in torch_model.mask_decoder.transformer.layers[
                    i
                ].mlp.parameters()
            ]
        )
        mb_model.mask_decoder.transformer.layers[i].layer_norm3.set_weights(
            [
                x.cpu().detach().numpy()
                for x in torch_model.mask_decoder.transformer.layers[
                    i
                ].norm3.parameters()
            ]
        )
        mb_model.mask_decoder.transformer.layers[
            i
        ].cross_attention_image_to_token.set_weights(
            [
                x.cpu().detach().numpy().T
                for x in torch_model.mask_decoder.transformer.layers[
                    i
                ].cross_attn_image_to_token.parameters()
            ]
        )
        mb_model.mask_decoder.transformer.layers[i].layer_norm4.set_weights(
            [
                x.cpu().detach().numpy()
                for x in torch_model.mask_decoder.transformer.layers[
                    i
                ].norm4.parameters()
            ]
        )
    mb_model.mask_decoder.transformer.final_attention_token_to_image.set_weights(
        [
            x.cpu().detach().numpy().T
            for x in torch_model.mask_decoder.transformer.final_attn_token_to_image.parameters()
        ]
    )
    mb_model.mask_decoder.transformer.final_layer_norm.set_weights(
        [
            x.cpu().detach().numpy()
            for x in torch_model.mask_decoder.transformer.norm_final_attn.parameters()
        ]
    )
    mb_model.mask_decoder.iou_token.set_weights(
        [
            x.cpu().detach().numpy()
            for x in torch_model.mask_decoder.iou_token.parameters()
        ]
    )
    mb_model.mask_decoder.mask_tokens.set_weights(
        [
            x.cpu().detach().numpy()
            for x in torch_model.mask_decoder.mask_tokens.parameters()
        ]
    )
    mb_model.mask_decoder.output_upscaling.set_weights(
        [
            x.permute(2, 3, 1, 0).cpu().detach().numpy()
            if x.ndim == 4
            else x.cpu().detach().numpy()
            for x in torch_model.mask_decoder.output_upscaling.parameters()
        ]
    )
    for i in range(mb_model.mask_decoder.num_mask_tokens):
        mb_model.mask_decoder.output_hypernetworks_mlps[i].set_weights(
            [
                x.cpu().detach().numpy().T
                for x in torch_model.mask_decoder.output_hypernetworks_mlps[
                    i
                ].parameters()
            ]
        )
    mb_model.mask_decoder.iou_prediction_head.set_weights(
        [
            x.cpu().detach().numpy().T
            for x in torch_model.mask_decoder.iou_prediction_head.parameters()
        ]
    )
    mb_model.image_encoder.patch_embed.set_weights(
        [
            x.permute(2, 3, 1, 0).cpu().detach().numpy()
            if x.ndim == 4
            else x.cpu().detach().numpy()
            for x in torch_model.image_encoder.patch_embed.parameters()
        ]
    )
    mb_model.image_encoder.pos_embed.assign(
        torch_model.image_encoder.pos_embed.cpu().detach().numpy()
    )
    for block in range(mb_model.image_encoder.transformer_blocks):
        block.layers[
            i
        ].layer_norm1.set_weights(
            [
                x.cpu().detach().numpy()
                for x in torch_model.image_encoder.blocks[i].norm1.parameters()
            ]
        )
        block.layers[
            i
        ].layer_norm2.set_weights(
            [
                x.cpu().detach().numpy()
                for x in torch_model.image_encoder.blocks[i].norm2.parameters()
            ]
        )
        block.layers[
            i
        ].attention.set_weights(
            [
                x.cpu().detach().numpy().T
                # This is kind of a hack but we won't need this script once we
                # publish the Keras models's weights.
                if x.shape[-1] == mb_model.image_encoder.embed_dim
                else x.cpu().detach().numpy()
                for x in torch_model.image_encoder.blocks[i].attn.parameters()
            ]
        )
        block.layers[
            i
        ].mlp_block.set_weights(
            [
                x.cpu().detach().numpy().T
                for x in torch_model.image_encoder.blocks[i].mlp.parameters()
            ]
        )
    mb_model.image_encoder.bottleneck.set_weights(
        [
            x.permute(2, 3, 1, 0).cpu().detach().numpy()
            if x.ndim == 4
            else x.cpu().detach().numpy()
            for x in torch_model.image_encoder.neck.parameters()
        ]
    )
