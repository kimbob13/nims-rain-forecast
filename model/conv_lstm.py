import torch
import torch.nn as nn

__all__ = ['EncoderForecaster', 'ConvLSTM']

class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, device):
        super(ConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_features = 4
        self.device = device

        self.padding = int((kernel_size - 1) / 2)

        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, x, h, c):
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):
        if self.Wci is None:
            self.Wci = nn.Parameter(torch.zeros(1, hidden, shape[0], shape[1], requires_grad=True)).to(self.device)
            self.Wcf = nn.Parameter(torch.zeros(1, hidden, shape[0], shape[1], requires_grad=True)).to(self.device)
            self.Wco = nn.Parameter(torch.zeros(1, hidden, shape[0], shape[1], requires_grad=True)).to(self.device)
        else:
            assert shape[0] == self.Wci.size()[2], 'Input Height Mismatched!'
            assert shape[1] == self.Wci.size()[3], 'Input Width Mismatched!'

        return ((torch.zeros(batch_size, hidden, shape[0], shape[1], requires_grad=True)).to(self.device),
                (torch.zeros(batch_size, hidden, shape[0], shape[1], requires_grad=True)).to(self.device))


class ConvLSTM(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    def __init__(self, input_channel, hidden_channel, kernel_size, seq_len, device=None):
        super(ConvLSTM, self).__init__()
        #self.input_channel = [input_channel] + hidden_channel
        self.hidden_channel = hidden_channel
        self.kernel_size = kernel_size
        #self.num_layers = len(hidden_channel)
        self.seq_len = seq_len

        self.cell = ConvLSTMCell(input_channel, hidden_channel, kernel_size, device)
        # for i in range(self.num_layers):
        #     name = 'cell{}'.format(i)
        #     cell = ConvLSTMCell(self.input_channel[i], self.hidden_channel[i], self.kernel_size, device)
        #     setattr(self, name, cell)
        #     self._all_layers.append(cell)

    def forward(self, inputs, states=None):
        """
        <Parameter>
        inputs [torch.tensor]: SNCHW format (S: seq_len, N: batch_size)
        """
        internal_state = None
        outputs = []

        for seq in range(self.seq_len):
            #print('[seq: {}] inputs type: {}'.format(seq, type(inputs)))
            x = inputs[seq, ...]

            if seq == 0:
                # Cell is initialized at the first sequence
                bsize, _, height, width = x.size()
                (h, c) = self.cell.init_hidden(batch_size=bsize, hidden=self.hidden_channel,
                                               shape=(height, width))

                if states != None:
                    # Use provided hidden states (by forecaster)
                    (h, c) = states

                internal_state = (h, c)

            # Do forward ConvLSTM Cell
            (h, c) = internal_state
            x, new_c = self.cell(x, h, c)
            internal_state = (x, new_c)

            outputs.append(x)

        #print('outputs shape: {}, outputs len: {}\ninternal state len: {}'.format((torch.stack(outputs)).shape, len(outputs), len(internal_state)))

        return torch.stack(outputs), (x, new_c)

class EncoderForecaster(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, seq_len, device=None):
        super(EncoderForecaster, self).__init__()

        assert isinstance(hidden_channels, list)

        self.encoder = nn.ModuleList([])
        self.encoder_conv = nn.ModuleList([])
        self.forecaster = nn.ModuleList([])
        self.forecaster_conv = nn.ModuleList([])
        
        self.num_blocks = len(hidden_channels)
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.seq_len = seq_len
        self.device = device
        
        for i in range(self.num_blocks):
            self.encoder.append(ConvLSTM(self.input_channels[i], hidden_channels[i],
                                         kernel_size, seq_len, device))
            self.forecaster.append(ConvLSTM(hidden_channels[-1 * (i + 1)], hidden_channels[-1 * (i + 1)],
                                            kernel_size, seq_len, device))

        for i in range(self.num_blocks):
            # e_conv = nn.Sequential()
            # e_conv.add_module("e_conv{}".format(i + 1),
            #                   nn.Conv2d(self.input_channels[i], self.hidden_channels[i],
            #                             kernel_size=3, stride=1, padding=1))
            # e_conv.add_module("e_relu{}".format(i + 1),
            #                   nn.ReLU(inplace=True))
            # self.encoder_conv.append(e_conv)

            f_conv = nn.Sequential()
            f_conv.add_module("f_conv{}".format(i + 1),
                              nn.Conv2d(self.hidden_channels[-1 * (i + 1)], self.input_channels[-1 * (i + 2)],
                                        kernel_size=3, stride=1, padding=1))
            f_conv.add_module("f_relu{}".format(i + 1),
                              nn.ReLU(inplace=True))
            self.forecaster_conv.append(f_conv)

    @property
    def name(self):
        return 'convlstm'

    def forward_one_conv(self, x, block_conv):
        seq_num, batch_size, input_channel, height, width = x.size()
        x = torch.reshape(x, (-1, input_channel, height, width))
        x = block_conv(x)
        x = torch.reshape(x, (seq_num, batch_size, x.size(1), x.size(2), x.size(3)))

        return x

    def forward(self, inputs):
        hidden_states = []

        # Encoder stage
        x = inputs
        for encoder in self.encoder:
            #x = self.forward_one_conv(x, e_conv)
            x, h_state = encoder(x)
            hidden_states.append(h_state)

        hidden_states = tuple(hidden_states)

        # Forecaster stage
        h, _ = hidden_states[-1] # h shape: NCHW (N: batch size, C: channels, H: height, W: width)
        x = torch.zeros((self.seq_len, h.shape[0], self.hidden_channels[-1],
                         h.shape[2], h.shape[3]), dtype=torch.float).to(self.device)
        for i, (forecaster, f_conv) in enumerate(zip(self.forecaster, self.forecaster_conv)):
            #print('[forecaster {}] x shape: {}, type: {}'.format(i, x.shape, type(x)))
            x, _ = forecaster(x, hidden_states[-1 * (i + 1)])
            x = self.forward_one_conv(x, f_conv)
             
        return x