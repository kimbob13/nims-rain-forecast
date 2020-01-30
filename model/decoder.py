import torch 
import torch.nn as nn
import torch.nn.functional as F
import time as tm        
        
class DecoderSTCNN(nn.Module):
    
    def __init__(self, layer_size, kernel_size, initial_filter_size, channels, dropout_rate, upsample=False):
        super(DecoderSTCNN, self).__init__()
        self.padding = kernel_size - 1
        self.upsample = upsample
        self.dropout_rate = dropout_rate
        self.conv_layers = nn.ModuleList()
        self.relu_layers = nn.ModuleList()
        self.batch_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        # Main layers
        temporal_kernel_size = [kernel_size, 1, 1]
        temporal_padding = [self.padding, 0, 0]
        out_channels = initial_filter_size
        in_channels = channels
        for i in range(layer_size):
            self.conv_layers.append(
                nn.Conv3d(in_channels=in_channels, out_channels=out_channels, 
                          kernel_size=temporal_kernel_size, padding=temporal_padding, bias=False)
            )
            self.relu_layers.append(nn.ReLU())
            self.batch_layers.append(nn.BatchNorm3d(out_channels))
            self.dropout_layers.append(nn.Dropout(dropout_rate))
            in_channels = out_channels
            
        # Upsample layer if enabled
        if upsample:
            self.upsample_conv = nn.ConvTranspose3d(out_channels, out_channels, kernel_size=temporal_kernel_size, 
                                                    stride=[3,1,1], padding=[1,0,0])
        else:
            self.downsample_conv = nn.Conv3d(out_channels, out_channels, kernel_size=temporal_kernel_size,
                                             stride=[1,1,1], padding=[1,0,0])

        # Final layer
        padding_final = [kernel_size // 2, 0, 0]
        self.conv_final = nn.Conv3d(in_channels=out_channels, out_channels=1, kernel_size=temporal_kernel_size, 
                                    padding=padding_final, bias=True)
        
        
    def learning_with_dropout(self, x):
        for conv, relu, batch, drop in zip(self.conv_layers, self.relu_layers, 
                                           self.batch_layers, self.dropout_layers):
            x = conv(x)[:,:,:-self.padding,:,:]
            x = drop(relu(batch(x)))
            
        return x
    
    def learning_without_dropout(self, x):
        for conv, relu, batch in zip(self.conv_layers, self.relu_layers, self.batch_layers):
            x = conv(x)[:,:,:-self.padding,:,:]
            x = relu(batch(x))
            
        return x
        
    def forward(self, input_):
        if self.dropout_rate > 0.:
            output = self.learning_with_dropout(input_)
        else:
            output = self.learning_without_dropout(input_)

        if self.upsample:
            #print('[decoder] input shape:', input_.shape)
            output_size = torch.randn(input_.shape[0], 1, input_.shape[2] + 20, 
                                      input_.shape[3], input_.shape[4]).size()
            #print('[decoder] output size:', output_size)
            output = self.upsample_conv(output, output_size=output_size)
        #else:
        #    output = self.downsample_conv(output)

        output = self.conv_final(output)
        return output
