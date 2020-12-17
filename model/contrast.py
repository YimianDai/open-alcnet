from __future__ import division
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from mxnet.gluon.nn import BatchNorm
from gluoncv.model_zoo.fcn import _FCNHead
from mxnet import nd
from gluoncv.model_zoo.cifarresnet import CIFARBasicBlockV1


def circ_shift(cen, shift):

    _, _, hei, wid = cen.shape

    ######## B1 #########
    # old: AD  =>  new: CB
    #      BC  =>       DA
    B1_NW = cen[:, :, shift:, shift:]          # B1_NW is cen's SE
    B1_NE = cen[:, :, shift:, :shift]      # B1_NE is cen's SW
    B1_SW = cen[:, :, :shift, shift:]      # B1_SW is cen's NE
    B1_SE = cen[:, :, :shift, :shift]          # B1_SE is cen's NW
    B1_N = nd.concat(B1_NW, B1_NE, dim=3)
    B1_S = nd.concat(B1_SW, B1_SE, dim=3)
    B1 = nd.concat(B1_N, B1_S, dim=2)

    ######## B2 #########
    # old: A  =>  new: B
    #      B  =>       A
    B2_N = cen[:, :, shift:, :]          # B2_N is cen's S
    B2_S = cen[:, :, :shift, :]      # B2_S is cen's N
    B2 = nd.concat(B2_N, B2_S, dim=2)

    ######## B3 #########
    # old: AD  =>  new: CB
    #      BC  =>       DA
    B3_NW = cen[:, :, shift:, wid-shift:]          # B3_NW is cen's SE
    B3_NE = cen[:, :, shift:, :wid-shift]      # B3_NE is cen's SW
    B3_SW = cen[:, :, :shift, wid-shift:]      # B3_SW is cen's NE
    B3_SE = cen[:, :, :shift, :wid-shift]          # B1_SE is cen's NW
    B3_N = nd.concat(B3_NW, B3_NE, dim=3)
    B3_S = nd.concat(B3_SW, B3_SE, dim=3)
    B3 = nd.concat(B3_N, B3_S, dim=2)

    ######## B4 #########
    # old: AB  =>  new: BA
    B4_W = cen[:, :, :, wid-shift:]          # B2_W is cen's E
    B4_E = cen[:, :, :, :wid-shift]          # B2_E is cen's S
    B4 = nd.concat(B4_W, B4_E, dim=3)

    ######## B5 #########
    # old: AD  =>  new: CB
    #      BC  =>       DA
    B5_NW = cen[:, :, hei-shift:, wid-shift:]          # B5_NW is cen's SE
    B5_NE = cen[:, :, hei-shift:, :wid-shift]      # B5_NE is cen's SW
    B5_SW = cen[:, :, :hei-shift, wid-shift:]      # B5_SW is cen's NE
    B5_SE = cen[:, :, :hei-shift, :wid-shift]          # B5_SE is cen's NW
    B5_N = nd.concat(B5_NW, B5_NE, dim=3)
    B5_S = nd.concat(B5_SW, B5_SE, dim=3)
    B5 = nd.concat(B5_N, B5_S, dim=2)

    ######## B6 #########
    # old: A  =>  new: B
    #      B  =>       A
    B6_N = cen[:, :, hei-shift:, :]          # B6_N is cen's S
    B6_S = cen[:, :, :hei-shift, :]      # B6_S is cen's N
    B6 = nd.concat(B6_N, B6_S, dim=2)

    ######## B7 #########
    # old: AD  =>  new: CB
    #      BC  =>       DA
    B7_NW = cen[:, :, hei-shift:, shift:]          # B7_NW is cen's SE
    B7_NE = cen[:, :, hei-shift:, :shift]      # B7_NE is cen's SW
    B7_SW = cen[:, :, :hei-shift, shift:]      # B7_SW is cen's NE
    B7_SE = cen[:, :, :hei-shift, :shift]          # B7_SE is cen's NW
    B7_N = nd.concat(B7_NW, B7_NE, dim=3)
    B7_S = nd.concat(B7_SW, B7_SE, dim=3)
    B7 = nd.concat(B7_N, B7_S, dim=2)

    ######## B8 #########
    # old: AB  =>  new: BA
    B8_W = cen[:, :, :, shift:]          # B8_W is cen's E
    B8_E = cen[:, :, :, :shift]          # B8_E is cen's S
    B8 = nd.concat(B8_W, B8_E, dim=3)

    return B1, B2, B3, B4, B5, B6, B7, B8


def cal_pcm(cen, shift):

    B1, B2, B3, B4, B5, B6, B7, B8 = circ_shift(cen, shift=shift)
    s1 = (B1 - cen) * (B5 - cen)
    s2 = (B2 - cen) * (B6 - cen)
    s3 = (B3 - cen) * (B7 - cen)
    s4 = (B4 - cen) * (B8 - cen)

    c12 = nd.minimum(s1, s2)
    c123 = nd.minimum(c12, s3)
    c1234 = nd.minimum(c123, s4)

    return c1234


class PCMNet(HybridBlock):
    def __init__(self, dilations=[1, 1, 2, 4, 8, 16], channels=16, classes=1, addstem=True,
                 maxpool=False, shift='xxx', **kwargs):
        super(PCMNet, self).__init__(**kwargs)

        self.shift = shift
        stem_width = int(channels // 2)
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            if addstem:
                self.features.add(nn.Conv2D(channels=stem_width, kernel_size=3, strides=2,
                                            padding=1, use_bias=False))
                self.features.add(nn.BatchNorm(in_channels=stem_width))
                self.features.add(nn.Activation('relu'))
                self.features.add(nn.Conv2D(channels=stem_width, kernel_size=3, strides=1,
                                            padding=1, use_bias=False))
                self.features.add(nn.BatchNorm(in_channels=stem_width))
                self.features.add(nn.Activation('relu'))
                self.features.add(nn.Conv2D(channels=stem_width*2, kernel_size=3, strides=1,
                                         padding=1, use_bias=False))
                self.features.add(nn.BatchNorm(in_channels=stem_width*2))
                self.features.add(nn.Activation('relu'))
            if maxpool:
                self.features.add(nn.MaxPool2D(pool_size=3, strides=2, padding=1))

            for i, dilation in enumerate(dilations):
                self.features.add(self._make_layer(
                    dilation=dilation, channels=channels, stage_index=i))

            self.head = _FCNHead(in_channels=channels, channels=classes)

    def _make_layer(self, dilation, channels, stage_index):
        layer = nn.HybridSequential(prefix='stage%d_' % stage_index)
        with layer.name_scope():

            layer.add(nn.Conv2D(channels=channels, kernel_size=3, dilation=dilation,
                                padding=dilation))
            layer.add(nn.BatchNorm())
            layer.add(nn.Activation('relu'))

        return layer

    def hybrid_forward(self, F, x):

        _, _, hei, wid = x.shape

        cen = self.features(x)
        c1234 = cal_pcm(cen, self.shift)
        x = self.head(c1234)

        out = F.contrib.BilinearResize2D(x, height=hei, width=wid)

        return out

    def evaluate(self, x):
        """evaluating network with inputs and targets"""
        return self.forward(x)


class MPCMNet(HybridBlock):
    def __init__(self, dilations=[1, 1, 2, 4, 8, 16], channels=16, classes=1, addstem=True,
                 maxpool=False, **kwargs):
        super(MPCMNet, self).__init__(**kwargs)

        stem_width = int(channels // 2)
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            if addstem:
                self.features.add(nn.Conv2D(channels=stem_width, kernel_size=3, strides=2,
                                            padding=1, use_bias=False))
                self.features.add(nn.BatchNorm(in_channels=stem_width))
                self.features.add(nn.Activation('relu'))
                self.features.add(nn.Conv2D(channels=stem_width, kernel_size=3, strides=1,
                                            padding=1, use_bias=False))
                self.features.add(nn.BatchNorm(in_channels=stem_width))
                self.features.add(nn.Activation('relu'))
                self.features.add(nn.Conv2D(channels=stem_width*2, kernel_size=3, strides=1,
                                         padding=1, use_bias=False))
                self.features.add(nn.BatchNorm(in_channels=stem_width*2))
                self.features.add(nn.Activation('relu'))
            if maxpool:
                self.features.add(nn.MaxPool2D(pool_size=3, strides=2, padding=1))

            for i, dilation in enumerate(dilations):
                self.features.add(self._make_layer(
                    dilation=dilation, channels=channels, stage_index=i))

            self.head = _FCNHead(in_channels=channels, channels=classes)

    def _make_layer(self, dilation, channels, stage_index):
        layer = nn.HybridSequential(prefix='stage%d_' % stage_index)
        with layer.name_scope():

            layer.add(nn.Conv2D(channels=channels, kernel_size=3, dilation=dilation,
                                padding=dilation))
            layer.add(nn.BatchNorm())
            layer.add(nn.Activation('relu'))

        return layer

    def hybrid_forward(self, F, x):

        _, _, hei, wid = x.shape

        cen = self.features(x)

        # pcm9 = self.cal_pcm(cen, shift=9)
        # pcm17 = self.cal_pcm(cen, shift=17)
        # pcm25 = self.cal_pcm(cen, shift=25)
        # pcm33 = self.cal_pcm(cen, shift=33)
        # mpcm = nd.maximum(nd.maximum(nd.maximum(pcm9, pcm17), pcm25), pcm33)

        pcm9 = cal_pcm(cen, shift=9)
        pcm13 = cal_pcm(cen, shift=13)
        pcm17 = cal_pcm(cen, shift=17)
        # pcm21 = self.cal_pcm(cen, shift=21)
        # mpcm = nd.maximum(nd.maximum(nd.maximum(pcm9, pcm13), pcm17), pcm21)
        mpcm = nd.maximum(nd.maximum(pcm9, pcm13), pcm17)

        x = self.head(mpcm)

        out = F.contrib.BilinearResize2D(x, height=hei, width=wid)

        return out

    def evaluate(self, x):
        """evaluating network with inputs and targets"""
        return self.forward(x)


class LayerwiseMPCMNet(HybridBlock):
    def __init__(self, dilations=[1, 1, 2, 4, 8, 16], channels=16, classes=1, addstem=True,
                 maxpool=False, **kwargs):
        super(LayerwiseMPCMNet, self).__init__(**kwargs)

        stem_width = int(channels // 2)
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            if addstem:
                self.features.add(nn.Conv2D(channels=stem_width, kernel_size=3, strides=2,
                                            padding=1, use_bias=False))
                self.features.add(nn.BatchNorm(in_channels=stem_width))
                self.features.add(nn.Activation('relu'))
                self.features.add(nn.Conv2D(channels=stem_width, kernel_size=3, strides=1,
                                            padding=1, use_bias=False))
                self.features.add(nn.BatchNorm(in_channels=stem_width))
                self.features.add(nn.Activation('relu'))
                self.features.add(nn.Conv2D(channels=stem_width*2, kernel_size=3, strides=1,
                                         padding=1, use_bias=False))
                self.features.add(nn.BatchNorm(in_channels=stem_width*2))
                self.features.add(nn.Activation('relu'))
            if maxpool:
                self.features.add(nn.MaxPool2D(pool_size=3, strides=2, padding=1))
            self.features.add(CalMPCM())

            for i, dilation in enumerate(dilations):
                self.features.add(self._make_layer(
                    dilation=dilation, channels=channels, stage_index=i))

            self.head = _FCNHead(in_channels=channels, channels=classes)

    def _make_layer(self, dilation, channels, stage_index):
        layer = nn.HybridSequential(prefix='stage%d_' % stage_index)
        with layer.name_scope():

            layer.add(nn.Conv2D(channels=channels, kernel_size=3, dilation=dilation,
                                padding=dilation))
            layer.add(nn.BatchNorm())
            layer.add(nn.Activation('relu'))
            layer.add(CalMPCM())

        return layer

    def hybrid_forward(self, F, x):

        _, _, hei, wid = x.shape

        x = self.features(x)
        x = self.head(x)

        out = F.contrib.BilinearResize2D(x, height=hei, width=wid)

        return out

    def evaluate(self, x):
        """evaluating network with inputs and targets"""
        return self.forward(x)


class PlainNet(HybridBlock):
    def __init__(self, dilations=[1, 1, 2, 4, 8, 16], channels=16, classes=1, addstem=True,
                 maxpool=False, **kwargs):
        super(PlainNet, self).__init__(**kwargs)

        stem_width = int(channels // 2)
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            if addstem:
                self.features.add(nn.Conv2D(channels=stem_width, kernel_size=3, strides=2,
                                            padding=1, use_bias=False))
                self.features.add(nn.BatchNorm(in_channels=stem_width))
                self.features.add(nn.Activation('relu'))
                self.features.add(nn.Conv2D(channels=stem_width, kernel_size=3, strides=1,
                                            padding=1, use_bias=False))
                self.features.add(nn.BatchNorm(in_channels=stem_width))
                self.features.add(nn.Activation('relu'))
                self.features.add(nn.Conv2D(channels=stem_width*2, kernel_size=3, strides=1,
                                         padding=1, use_bias=False))
                self.features.add(nn.BatchNorm(in_channels=stem_width*2))
                self.features.add(nn.Activation('relu'))
            if maxpool:
                self.features.add(nn.MaxPool2D(pool_size=3, strides=2, padding=1))

            for i, dilation in enumerate(dilations):
                self.features.add(self._make_layer(
                    dilation=dilation, channels=channels, stage_index=i))

            self.head = _FCNHead(in_channels=channels, channels=classes)

    def _make_layer(self, dilation, channels, stage_index):
        layer = nn.HybridSequential(prefix='stage%d_' % stage_index)
        with layer.name_scope():

            layer.add(nn.Conv2D(channels=channels, kernel_size=3, dilation=dilation,
                                padding=dilation))
            layer.add(nn.BatchNorm())
            layer.add(nn.Activation('relu'))

        return layer

    def hybrid_forward(self, F, x):

        _, _, hei, wid = x.shape

        cen = self.features(x)

        x = self.head(cen)

        out = F.contrib.BilinearResize2D(x, height=hei, width=wid)

        return out

    def evaluate(self, x):
        """evaluating network with inputs and targets"""
        return self.forward(x)


class CalMPCM(HybridBlock):
    def __init__(self, **kwargs):
        super(CalMPCM, self).__init__(**kwargs)

    def hybrid_forward(self, F, x):

        pcm9 = cal_pcm(x, shift=9)
        pcm13 = cal_pcm(x, shift=13)
        pcm17 = cal_pcm(x, shift=17)
        mpcm = nd.maximum(nd.maximum(pcm9, pcm13), pcm17)

        return mpcm

    def evaluate(self, x):
        """evaluating network with inputs and targets"""
        return self.forward(x)


class MPCMResNetFPN(HybridBlock):
    def __init__(self, layers, channels, shift=3, pyramid_mode='xxx', scale_mode='xxx',
                 pyramid_fuse='xxx', r=2, classes=1, norm_layer=BatchNorm, norm_kwargs=None,
                 **kwargs):
        super(MPCMResNetFPN, self).__init__(**kwargs)

        self.layer_num = len(layers)
        with self.name_scope():

            self.r = r
            self.shift = shift
            self.pyramid_mode = pyramid_mode
            self.scale_mode = scale_mode
            self.pyramid_fuse = pyramid_fuse

            stem_width = int(channels[0])
            self.stem = nn.HybridSequential(prefix='stem')
            self.stem.add(norm_layer(scale=False, center=False,
                                     **({} if norm_kwargs is None else norm_kwargs)))
            self.stem.add(nn.Conv2D(channels=stem_width*2, kernel_size=3, strides=1,
                                     padding=1, use_bias=False))
            self.stem.add(norm_layer(in_channels=stem_width*2))
            self.stem.add(nn.Activation('relu'))

            self.head = _FCNHead(in_channels=channels[1], channels=classes)

            self.layer1 = self._make_layer(block=CIFARBasicBlockV1, layers=layers[0],
                                           channels=channels[1], stride=1, stage_index=1,
                                           in_channels=channels[1])

            self.layer2 = self._make_layer(block=CIFARBasicBlockV1, layers=layers[1],
                                           channels=channels[2], stride=2, stage_index=2,
                                           in_channels=channels[1])

            self.layer3 = self._make_layer(block=CIFARBasicBlockV1, layers=layers[2],
                                           channels=channels[3], stride=2, stage_index=3,
                                           in_channels=channels[2])

            if pyramid_mode == 'Dec':

                self.dec_c2 = nn.HybridSequential(prefix='dec_c2')
                self.dec_c2.add(nn.Conv2D(channels=channels[1], kernel_size=1, strides=1,
                                         padding=0, use_bias=False))
                self.dec_c2.add(norm_layer(in_channels=channels[1]))
                self.dec_c2.add(nn.Activation('relu'))

                self.dec_c3 = nn.HybridSequential(prefix='dec_c3')
                self.dec_c3.add(nn.Conv2D(channels=channels[1], kernel_size=1, strides=1,
                                         padding=0, use_bias=False))
                self.dec_c3.add(norm_layer(in_channels=channels[1]))
                self.dec_c3.add(nn.Activation('relu'))

                # if self.scale_mode == 'Selective':
                #     # self.fuse_mpcm_c1 = GlobMPCMFuse(channels=channels[1])
                #     # self.fuse_mpcm_c2 = GlobMPCMFuse(channels=channels[2])
                #     # self.fuse_mpcm_c3 = GlobMPCMFuse(channels=channels[3])
                #
                #     self.fuse_mpcm_c1 = LocalMPCMFuse(channels=channels[1])
                #     self.fuse_mpcm_c2 = LocalMPCMFuse(channels=channels[2])
                #     self.fuse_mpcm_c3 = LocalMPCMFuse(channels=channels[3])

                if self.scale_mode == 'biglobal':
                    self.fuse_mpcm_c1 = BiGlobal_MPCMFuse(channels=channels[1])
                    self.fuse_mpcm_c2 = BiGlobal_MPCMFuse(channels=channels[2])
                    self.fuse_mpcm_c3 = BiGlobal_MPCMFuse(channels=channels[3])
                elif self.scale_mode == 'bilocal':
                    self.fuse_mpcm_c1 = BiLocal_MPCMFuse(channels=channels[1])
                    self.fuse_mpcm_c2 = BiLocal_MPCMFuse(channels=channels[2])
                    self.fuse_mpcm_c3 = BiLocal_MPCMFuse(channels=channels[3])
                elif self.scale_mode == 'add':
                    self.fuse_mpcm_c1 = Add_MPCMFuse(channels=channels[1])
                    self.fuse_mpcm_c2 = Add_MPCMFuse(channels=channels[2])
                    self.fuse_mpcm_c3 = Add_MPCMFuse(channels=channels[3])
                elif self.scale_mode == 'globalsk':
                    self.fuse_mpcm_c1 = GlobalSK_MPCMFuse(channels=channels[1])
                    self.fuse_mpcm_c2 = GlobalSK_MPCMFuse(channels=channels[2])
                    self.fuse_mpcm_c3 = GlobalSK_MPCMFuse(channels=channels[3])
                elif self.scale_mode == 'localsk':
                    self.fuse_mpcm_c1 = LocalSK_MPCMFuse(channels=channels[1])
                    self.fuse_mpcm_c2 = LocalSK_MPCMFuse(channels=channels[2])
                    self.fuse_mpcm_c3 = LocalSK_MPCMFuse(channels=channels[3])

                if self.pyramid_fuse == 'globalsk':
                    self.globalsk_fpn_2 = GlobalSK_FPNFuse(channels=channels[1], r=self.r)
                    self.globalsk_fpn_1 = GlobalSK_FPNFuse(channels=channels[1], r=self.r)
                elif self.pyramid_fuse == 'localsk':
                    self.localsk_fpn_2 = LocalSK_FPNFuse(channels=channels[1], r=self.r)
                    self.localsk_fpn_1 = LocalSK_FPNFuse(channels=channels[1], r=self.r)
                elif self.pyramid_fuse == 'mutualglobalsk':
                    self.mutualglobalsk_fpn_2 = MutualSKGlobal_FPNFuse(channels=channels[1])
                    self.mutualglobalsk_fpn_1 = MutualSKGlobal_FPNFuse(channels=channels[1])
                elif self.pyramid_fuse == 'mutuallocalsk':
                    self.mutuallocalsk_fpn_2 = MutualSKLocal_FPNFuse(channels=channels[1])
                    self.mutuallocalsk_fpn_1 = MutualSKLocal_FPNFuse(channels=channels[1])
                elif self.pyramid_fuse == 'bilocal':
                    self.bilocal_fpn_2 = BiLocal_FPNFuse(channels=channels[1])
                    self.bilocal_fpn_1 = BiLocal_FPNFuse(channels=channels[1])
                elif self.pyramid_fuse == 'biglobal':
                    self.biglobal_fpn_2 = BiGlobal_FPNFuse(channels=channels[1])
                    self.biglobal_fpn_1 = BiGlobal_FPNFuse(channels=channels[1])
                elif self.pyramid_fuse == 'remo':
                    self.remo_fpn_2 = ReMo_FPNFuse(channels=channels[1])
                    self.remo_fpn_1 = ReMo_FPNFuse(channels=channels[1])
                elif self.pyramid_fuse == 'localremo':
                    self.localremo_fpn_2 = ReMo_FPNFuse(channels=channels[1])
                    self.localremo_fpn_1 = ReMo_FPNFuse(channels=channels[1])
                elif self.pyramid_fuse == 'asymbi':
                    self.asymbi_fpn_2 = AsymBi_FPNFuse(channels=channels[1])
                    self.asymbi_fpn_1 = AsymBi_FPNFuse(channels=channels[1])
                elif self.pyramid_fuse == 'topdownlocal':
                    self.topdownlocal_fpn_2 = TopDownLocal_FPNFuse(channels=channels[1])
                    self.topdownlocal_fpn_1 = TopDownLocal_FPNFuse(channels=channels[1])
                elif self.pyramid_fuse == 'bottomuplocal':
                    self.bottomuplocal_fpn_2 = BottomUpLocal_FPNFuse(channels=channels[1])
                    self.bottomuplocal_fpn_1 = BottomUpLocal_FPNFuse(channels=channels[1])
                elif self.pyramid_fuse == 'bottomupglobal':
                    self.bottomupglobal_fpn_2 = BottomUpGlobal_FPNFuse(channels=channels[1])
                    self.bottomupglobal_fpn_1 = BottomUpGlobal_FPNFuse(channels=channels[1])


            # elif pyramid_mode == 'Inc':
            #
            #     self.inc_c2 = nn.HybridSequential(prefix='inc_c2')
            #     self.inc_c2.add(nn.Conv2D(channels=channels[3], kernel_size=1, strides=1,
            #                              padding=0, use_bias=False))
            #     self.inc_c2.add(norm_layer(in_channels=channels[-1]))
            #     self.inc_c2.add(nn.Activation('relu'))
            #
            #     self.inc_c1 = nn.HybridSequential(prefix='inc_c1')
            #     self.inc_c1.add(nn.Conv2D(channels=channels[3], kernel_size=1, strides=1,
            #                              padding=0, use_bias=False))
            #     self.inc_c1.add(norm_layer(in_channels=channels[-1]))
            #     self.inc_c1.add(nn.Activation('relu'))
            # else:
            #     raise ValueError("unknown pyramid_mode")

    def _make_layer(self, block, layers, channels, stride, stage_index, in_channels=0,
                    norm_layer=BatchNorm, norm_kwargs=None):
        layer = nn.HybridSequential(prefix='stage%d_'%stage_index)
        with layer.name_scope():
            downsample = (channels != in_channels) or (stride != 1)
            layer.add(block(channels, stride, downsample, in_channels=in_channels,
                            prefix='', norm_layer=norm_layer, norm_kwargs=norm_kwargs))
            for _ in range(layers-1):
                layer.add(block(channels, 1, False, in_channels=channels, prefix='',
                                norm_layer=norm_layer, norm_kwargs=norm_kwargs))
        return layer

    def hybrid_forward(self, F, x):

        _, _, orig_hei, orig_wid = x.shape
        x = self.stem(x)      # sub 2
        c1 = self.layer1(x)   # sub 2
        _, _, c1_hei, c1_wid = c1.shape
        c2 = self.layer2(c1)  # sub 4
        _, _, c2_hei, c2_wid = c2.shape
        c3 = self.layer3(c2)  # sub 8
        _, _, c3_hei, c3_wid = c3.shape

        # 1. upsampling(c3) -> c3PCM   # size: sub 4

        # c3 -> c3PCM
        # 2. pwconv(c2) -> c2PCM       # size: sub 4
        # 3. upsampling(c3PCM + c2PCM) # size: sub 2
        # 4. pwconv(c1) -> c1PCM       # size: sub 2
        # 5. upsampling(upsampling(c3PCM + c2PCM)) + c1PCM
        # 6. upsampling(upsampling(c3PCM + c2PCM)) + c1PCM

        if self.pyramid_mode == 'Dec':
            if self.scale_mode == 'Single':
                c3pcm = cal_pcm(c3, shift=self.shift)  # sub 8, 64
            elif self.scale_mode == 'Multiple':
                c3pcm = self.cal_mpcm(c3)  # sub 8, 64
            elif self.scale_mode == 'biglobal':
                c3pcm = self.fuse_mpcm_c3(c3)  # sub 8, 64
            elif self.scale_mode == 'bilocal':
                c3pcm = self.fuse_mpcm_c3(c3)  # sub 8, 64
            elif self.scale_mode == 'add':
                c3pcm = self.fuse_mpcm_c3(c3)  # sub 8, 64
            elif self.scale_mode == 'globalsk':
                c3pcm = self.fuse_mpcm_c3(c3)  # sub 8, 64
            elif self.scale_mode == 'localsk':
                c3pcm = self.fuse_mpcm_c3(c3)  # sub 8, 64
            else:
                raise ValueError("unknow self.scale_mode")
            c3pcm = self.dec_c3(c3pcm)                  # sub 8, 16
            up_c3pcm = F.contrib.BilinearResize2D(c3pcm, height=c2_hei, width=c2_wid)  # sub 4, 16

            if self.scale_mode == 'Single':
                c2pcm = cal_pcm(c2, shift=self.shift)  # sub 4, 32
            elif self.scale_mode == 'Multiple':
                c2pcm = self.cal_mpcm(c2)  # sub 4, 32
            elif self.scale_mode == 'biglobal':
                c2pcm = self.fuse_mpcm_c2(c2)  # sub 4, 32
            elif self.scale_mode == 'bilocal':
                c2pcm = self.fuse_mpcm_c2(c2)  # sub 4, 32
            elif self.scale_mode == 'add':
                c2pcm = self.fuse_mpcm_c2(c2)  # sub 4, 32
            elif self.scale_mode == 'globalsk':
                c2pcm = self.fuse_mpcm_c2(c2)  # sub 4, 32
            elif self.scale_mode == 'localsk':
                c2pcm = self.fuse_mpcm_c2(c2)  # sub 4, 32
            else:
                raise ValueError("unknow self.scale_mode")
            c2pcm = self.dec_c2(c2pcm)                  # sub 4, 16

            if self.pyramid_fuse == 'add':
                c23pcm = up_c3pcm + c2pcm                   # sub 4, 16
            elif self.pyramid_fuse == 'max':
                c23pcm = nd.maximum(up_c3pcm, c2pcm)                  # sub 4, 16
            elif self.pyramid_fuse == 'bilocal':
                c23pcm = self.bilocal_fpn_2(up_c3pcm, c2pcm)
            elif self.pyramid_fuse == 'biglobal':
                c23pcm = self.biglobal_fpn_2(up_c3pcm, c2pcm)
            elif self.pyramid_fuse == 'globalsk':
                c23pcm = self.globalsk_fpn_2(up_c3pcm, c2pcm)
            elif self.pyramid_fuse == 'localsk':
                c23pcm = self.localsk_fpn_2(up_c3pcm, c2pcm)
            elif self.pyramid_fuse == 'mutualglobalsk':
                c23pcm = self.mutualglobalsk_fpn_2(up_c3pcm, c2pcm)
            elif self.pyramid_fuse == 'mutuallocalsk':
                c23pcm = self.mutuallocalsk_fpn_2(up_c3pcm, c2pcm)
            elif self.pyramid_fuse == 'remo':
                c23pcm = self.remo_fpn_2(up_c3pcm, c2pcm)
            elif self.pyramid_fuse == 'localremo':
                c23pcm = self.localremo_fpn_2(up_c3pcm, c2pcm)
            elif self.pyramid_fuse == 'asymbi':
                c23pcm = self.asymbi_fpn_2(up_c3pcm, c2pcm)
            elif self.pyramid_fuse == 'topdownlocal':
                c23pcm = self.topdownlocal_fpn_2(up_c3pcm, c2pcm)
            elif self.pyramid_fuse == 'bottomuplocal':
                c23pcm = self.bottomuplocal_fpn_2(up_c3pcm, c2pcm)
            elif self.pyramid_fuse == 'bottomupglobal':
                c23pcm = self.bottomupglobal_fpn_2(up_c3pcm, c2pcm)
            else:
                raise ValueError("unknow self.scale_mode")

            up_c23pcm = F.contrib.BilinearResize2D(c23pcm, height=c1_hei, width=c1_wid)  # sub 2, 16

            if self.scale_mode == 'Single':
                c1pcm = cal_pcm(c1, shift=self.shift)  # sub 2, 16
            elif self.scale_mode == 'Multiple':
                c1pcm = self.cal_mpcm(c1)  # sub 2, 16
            elif self.scale_mode == 'biglobal':
                c1pcm = self.fuse_mpcm_c1(c1)  # sub 2, 16
            elif self.scale_mode == 'bilocal':
                c1pcm = self.fuse_mpcm_c1(c1)  # sub 2, 16
            elif self.scale_mode == 'add':
                c1pcm = self.fuse_mpcm_c1(c1)  # sub 2, 16
            elif self.scale_mode == 'globalsk':
                c1pcm = self.fuse_mpcm_c1(c1)  # sub 2, 16
            elif self.scale_mode == 'localsk':
                c1pcm = self.fuse_mpcm_c1(c1)  # sub 2, 16
            else:
                raise ValueError("unknow self.scale_mode")

            if self.pyramid_fuse == 'add':
                out = up_c23pcm + c1pcm
            elif self.pyramid_fuse == 'max':
                out = nd.maximum(up_c23pcm, c1pcm)
            elif self.pyramid_fuse == 'bilocal':
                out = self.bilocal_fpn_1(up_c23pcm, c1pcm)
            elif self.pyramid_fuse == 'biglobal':
                out = self.biglobal_fpn_1(up_c23pcm, c1pcm)
            elif self.pyramid_fuse == 'globalsk':
                out = self.globalsk_fpn_1(up_c23pcm, c1pcm)
            elif self.pyramid_fuse == 'localsk':
                out = self.localsk_fpn_1(up_c23pcm, c1pcm)
            elif self.pyramid_fuse == 'mutualglobalsk':
                out = self.mutualglobalsk_fpn_1(up_c23pcm, c1pcm)
            elif self.pyramid_fuse == 'mutuallocalsk':
                out = self.mutuallocalsk_fpn_1(up_c23pcm, c1pcm)
            elif self.pyramid_fuse == 'remo':
                out = self.remo_fpn_1(up_c23pcm, c1pcm)
            elif self.pyramid_fuse == 'localremo':
                out = self.localremo_fpn_1(up_c23pcm, c1pcm)
            elif self.pyramid_fuse == 'asymbi':
                out = self.asymbi_fpn_1(up_c23pcm, c1pcm)
            elif self.pyramid_fuse == 'topdownlocal':
                out = self.topdownlocal_fpn_1(up_c23pcm, c1pcm)
            elif self.pyramid_fuse == 'bottomuplocal':
                out = self.bottomuplocal_fpn_1(up_c23pcm, c1pcm)
            elif self.pyramid_fuse == 'bottomupglobal':
                out = self.bottomupglobal_fpn_1(up_c23pcm, c1pcm)
            else:
                raise ValueError("unknown self.pyramid_fuse")

        elif self.pyramid_mode == 'Inc':

            c3pcm = cal_pcm(c3, shift=self.shift)
            up_c3pcm = F.contrib.BilinearResize2D(c3pcm, height=c2_hei, width=c2_wid) # sub 4, 64

            inc_c2 = self.inc_c2(c2)               # sub 4, 64
            c2pcm = cal_pcm(inc_c2, shift=self.shift)

            c23pcm = up_c3pcm + c2pcm              # sub 4, 64

            up_c23pcm = F.contrib.BilinearResize2D(c23pcm, height=c1_hei, width=c1_wid)  # sub 2, 64
            inc_c1 = self.inc_c1(c1)               # sub 2, 64
            c1pcm = cal_pcm(inc_c1, shift=self.shift)

            out = up_c23pcm + c1pcm              # sub 2, 64

        pred = self.head(out)
        out = F.contrib.BilinearResize2D(pred, height=orig_hei, width=orig_wid)

        return out

    def evaluate(self, x):
        """evaluating network with inputs and targets"""
        return self.forward(x)

    def cal_mpcm(self, cen):
        # pcm11 = cal_pcm(cen, shift=11)
        pcm13 = cal_pcm(cen, shift=13)
        pcm17 = cal_pcm(cen, shift=17)
        mpcm = nd.maximum(pcm13, pcm17)
        # mpcm = nd.maximum(pcm11, nd.maximum(pcm13, pcm17))

        return mpcm


#### MPCM Fuse


class Add_MPCMFuse(HybridBlock):
    def __init__(self, channels=64):
        super(Add_MPCMFuse, self).__init__()

        with self.name_scope():

            self.bn1 = nn.BatchNorm()
            self.bn2 = nn.BatchNorm()

    def hybrid_forward(self, F, cen):

        pcm13 = cal_pcm(cen, shift=13)
        pcm17 = cal_pcm(cen, shift=17)

        pcm13 = self.bn1(pcm13)
        pcm17 = self.bn2(pcm17)

        xo = pcm13 + pcm17

        return xo


class LocalSK_MPCMFuse(HybridBlock):
    def __init__(self, channels=64):
        super(LocalSK_MPCMFuse, self).__init__()
        inter_channels = int(channels // 2)

        with self.name_scope():

            self.local_att = nn.HybridSequential(prefix='local_att')
            self.local_att.add(nn.Conv2D(inter_channels, kernel_size=1, strides=1, padding=0))
            self.local_att.add(nn.BatchNorm())
            self.local_att.add(nn.Activation('relu'))
            self.local_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.local_att.add(nn.BatchNorm())
            self.local_att.add(nn.Activation('sigmoid'))

    def hybrid_forward(self, F, cen):

        pcm13 = cal_pcm(cen, shift=13)
        pcm17 = cal_pcm(cen, shift=17)

        xa = pcm13 + pcm17
        # xa = cen
        wei = self.local_att(xa)

        xo = 2 * F.broadcast_mul(pcm13, wei) + 2 * F.broadcast_mul(pcm17, 1-wei)

        return xo


class GlobalSK_MPCMFuse(HybridBlock):
    def __init__(self, channels=64):
        super(GlobalSK_MPCMFuse, self).__init__()
        inter_channels = int(channels // 2)

        with self.name_scope():

            self.global_att = nn.HybridSequential(prefix='global_att')
            self.global_att.add(nn.GlobalAvgPool2D())
            self.global_att.add(nn.Conv2D(inter_channels, kernel_size=1, strides=1, padding=0))
            self.global_att.add(nn.BatchNorm())
            self.global_att.add(nn.Activation('relu'))
            self.global_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.global_att.add(nn.BatchNorm())
            self.global_att.add(nn.Activation('sigmoid'))

    def hybrid_forward(self, F, cen):

        pcm13 = cal_pcm(cen, shift=13)
        pcm17 = cal_pcm(cen, shift=17)

        xa = pcm13 + pcm17
        wei = self.global_att(xa)

        xo = 2 * F.broadcast_mul(pcm13, wei) + 2 * F.broadcast_mul(pcm17, 1-wei)

        return xo


class BiLocal_MPCMFuse(HybridBlock):
    def __init__(self, channels=64):
        super(BiLocal_MPCMFuse, self).__init__()
        inter_channels = int(channels // 2)

        with self.name_scope():

            self.bn1 = nn.BatchNorm()
            self.bn2 = nn.BatchNorm()

            self.topdown_att = nn.HybridSequential(prefix='topdown_att')
            self.topdown_att.add(nn.Conv2D(inter_channels, kernel_size=1, strides=1, padding=0))
            self.topdown_att.add(nn.BatchNorm())
            self.topdown_att.add(nn.Activation('relu'))
            self.topdown_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.topdown_att.add(nn.BatchNorm())
            self.topdown_att.add(nn.Activation('sigmoid'))

            self.bottomup_att = nn.HybridSequential(prefix='bottomup_att')
            self.bottomup_att.add(nn.Conv2D(inter_channels, kernel_size=1, strides=1, padding=0))
            self.bottomup_att.add(nn.BatchNorm())
            self.bottomup_att.add(nn.Activation('relu'))
            self.bottomup_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.bottomup_att.add(nn.BatchNorm())
            self.bottomup_att.add(nn.Activation('sigmoid'))

    def hybrid_forward(self, F, cen):

        pcm13 = cal_pcm(cen, shift=13)
        pcm17 = cal_pcm(cen, shift=17)

        pcm13 = self.bn1(pcm13)
        pcm17 = self.bn2(pcm17)

        topdown_wei = self.topdown_att(pcm17)
        bottomup_wei = self.bottomup_att(pcm13)

        xo = F.broadcast_mul(topdown_wei, pcm13) + F.broadcast_mul(bottomup_wei, pcm17)

        return xo


class BiGlobal_MPCMFuse(HybridBlock):
    def __init__(self, channels=64):
        super(BiGlobal_MPCMFuse, self).__init__()
        inter_channels = int(channels // 2)

        with self.name_scope():

            self.bn1 = nn.BatchNorm()
            self.bn2 = nn.BatchNorm()

            self.topdown_att = nn.HybridSequential(prefix='topdown_att')
            self.topdown_att.add(nn.GlobalAvgPool2D())
            self.topdown_att.add(nn.Conv2D(inter_channels, kernel_size=1, strides=1, padding=0))
            self.topdown_att.add(nn.BatchNorm())
            self.topdown_att.add(nn.Activation('relu'))
            self.topdown_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.topdown_att.add(nn.BatchNorm())
            self.topdown_att.add(nn.Activation('sigmoid'))

            self.bottomup_att = nn.HybridSequential(prefix='bottomup_att')
            self.bottomup_att.add(nn.GlobalAvgPool2D())
            self.bottomup_att.add(nn.Conv2D(inter_channels, kernel_size=1, strides=1, padding=0))
            self.bottomup_att.add(nn.BatchNorm())
            self.bottomup_att.add(nn.Activation('relu'))
            self.bottomup_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.bottomup_att.add(nn.BatchNorm())
            self.bottomup_att.add(nn.Activation('sigmoid'))

    def hybrid_forward(self, F, cen):

        pcm13 = cal_pcm(cen, shift=13)
        pcm17 = cal_pcm(cen, shift=17)

        pcm13 = self.bn1(pcm13)
        pcm17 = self.bn2(pcm17)

        topdown_wei = self.topdown_att(pcm17)
        bottomup_wei = self.bottomup_att(pcm13)

        xo = F.broadcast_mul(topdown_wei, pcm13) + F.broadcast_mul(bottomup_wei, pcm17)

        return xo


### FPN Fuse

class BiGlobal_FPNFuse(HybridBlock):
    def __init__(self, channels=64):
        super(BiGlobal_FPNFuse, self).__init__()
        inter_channels = int(channels // 2)

        with self.name_scope():

            self.bn1 = nn.BatchNorm()
            self.bn2 = nn.BatchNorm()

            self.topdown_att = nn.HybridSequential(prefix='topdown_att')
            self.topdown_att.add(nn.GlobalAvgPool2D())
            self.topdown_att.add(nn.Conv2D(inter_channels, kernel_size=1, strides=1, padding=0))
            self.topdown_att.add(nn.BatchNorm())
            self.topdown_att.add(nn.Activation('relu'))
            self.topdown_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.topdown_att.add(nn.BatchNorm())
            self.topdown_att.add(nn.Activation('sigmoid'))

            self.bottomup_att = nn.HybridSequential(prefix='bottomup_att')
            self.bottomup_att.add(nn.GlobalAvgPool2D())
            self.bottomup_att.add(nn.Conv2D(inter_channels, kernel_size=1, strides=1, padding=0))
            self.bottomup_att.add(nn.BatchNorm())
            self.bottomup_att.add(nn.Activation('relu'))
            self.bottomup_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.bottomup_att.add(nn.BatchNorm())
            self.bottomup_att.add(nn.Activation('sigmoid'))

    def hybrid_forward(self, F, x, residual):

        x = self.bn1(x)
        residual = self.bn2(residual)

        topdown_wei = self.topdown_att(x)
        bottomup_wei = self.bottomup_att(residual)

        xo = F.broadcast_mul(topdown_wei, residual) + F.broadcast_mul(bottomup_wei, x)

        return xo


class BiLocal_FPNFuse(HybridBlock):
    def __init__(self, channels=64):
        super(BiLocal_FPNFuse, self).__init__()
        inter_channels = int(channels // 2)

        with self.name_scope():

            self.bn1 = nn.BatchNorm()
            self.bn2 = nn.BatchNorm()

            self.topdown_att = nn.HybridSequential(prefix='topdown_att')
            self.topdown_att.add(nn.Conv2D(inter_channels, kernel_size=1, strides=1, padding=0))
            self.topdown_att.add(nn.BatchNorm())
            self.topdown_att.add(nn.Activation('relu'))
            self.topdown_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.topdown_att.add(nn.BatchNorm())
            self.topdown_att.add(nn.Activation('sigmoid'))

            self.bottomup_att = nn.HybridSequential(prefix='bottomup_att')
            self.bottomup_att.add(nn.Conv2D(inter_channels, kernel_size=1, strides=1, padding=0))
            self.bottomup_att.add(nn.BatchNorm())
            self.bottomup_att.add(nn.Activation('relu'))
            self.bottomup_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.bottomup_att.add(nn.BatchNorm())
            self.bottomup_att.add(nn.Activation('sigmoid'))

    def hybrid_forward(self, F, x, residual):

        x = self.bn1(x)
        residual = self.bn2(residual)

        topdown_wei = self.topdown_att(x)
        bottomup_wei = self.bottomup_att(residual)

        xo = F.broadcast_mul(topdown_wei, residual) + F.broadcast_mul(bottomup_wei, x)

        return xo


class AsymBi_FPNFuse(HybridBlock):
    def __init__(self, channels=64):
        super(AsymBi_FPNFuse, self).__init__()
        inter_channels = int(channels // 2)

        with self.name_scope():

            self.bn1 = nn.BatchNorm()
            self.bn2 = nn.BatchNorm()

            self.topdown_att = nn.HybridSequential(prefix='topdown_att')
            self.topdown_att.add(nn.GlobalAvgPool2D())
            self.topdown_att.add(nn.Conv2D(inter_channels, kernel_size=1, strides=1, padding=0))
            self.topdown_att.add(nn.BatchNorm())
            self.topdown_att.add(nn.Activation('relu'))
            self.topdown_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.topdown_att.add(nn.BatchNorm())
            self.topdown_att.add(nn.Activation('sigmoid'))

            self.bottomup_att = nn.HybridSequential(prefix='bottomup_att')
            self.bottomup_att.add(nn.Conv2D(inter_channels, kernel_size=1, strides=1, padding=0))
            self.bottomup_att.add(nn.BatchNorm())
            self.bottomup_att.add(nn.Activation('relu'))
            self.bottomup_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.bottomup_att.add(nn.BatchNorm())
            self.bottomup_att.add(nn.Activation('sigmoid'))

    def hybrid_forward(self, F, x, residual):

        x = self.bn1(x)
        residual = self.bn2(residual)

        topdown_wei = self.topdown_att(x)
        bottomup_wei = self.bottomup_att(residual)

        xo = F.broadcast_mul(topdown_wei, residual) + F.broadcast_mul(bottomup_wei, x)

        return xo


class BottomUpLocal_FPNFuse(HybridBlock):
    def __init__(self, channels=64):
        super(BottomUpLocal_FPNFuse, self).__init__()
        inter_channels = int(channels // 1)

        with self.name_scope():

            self.bn1 = nn.BatchNorm()
            self.bn2 = nn.BatchNorm()

            self.bottomup_att = nn.HybridSequential(prefix='bottomup_att')
            self.bottomup_att.add(nn.Conv2D(inter_channels, kernel_size=1, strides=1, padding=0))
            self.bottomup_att.add(nn.BatchNorm())
            self.bottomup_att.add(nn.Activation('relu'))
            self.bottomup_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.bottomup_att.add(nn.BatchNorm())
            self.bottomup_att.add(nn.Activation('sigmoid'))

    def hybrid_forward(self, F, x, residual):

        x = self.bn1(x)
        residual = self.bn2(residual)

        bottomup_wei = self.bottomup_att(residual)

        xo = F.broadcast_mul(bottomup_wei, x) + residual

        return xo


class BottomUpGlobal_FPNFuse(HybridBlock):
    def __init__(self, channels=64):
        super(BottomUpGlobal_FPNFuse, self).__init__()
        inter_channels = int(channels // 1)

        with self.name_scope():

            self.bn1 = nn.BatchNorm()
            self.bn2 = nn.BatchNorm()

            self.bottomup_att = nn.HybridSequential(prefix='bottomup_att')
            self.bottomup_att.add(nn.GlobalAvgPool2D())
            self.bottomup_att.add(nn.Conv2D(inter_channels, kernel_size=1, strides=1, padding=0))
            self.bottomup_att.add(nn.BatchNorm())
            self.bottomup_att.add(nn.Activation('relu'))
            self.bottomup_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.bottomup_att.add(nn.BatchNorm())
            self.bottomup_att.add(nn.Activation('sigmoid'))

    def hybrid_forward(self, F, x, residual):

        x = self.bn1(x)
        residual = self.bn2(residual)

        bottomup_wei = self.bottomup_att(residual)

        xo = F.broadcast_mul(bottomup_wei, x) + residual

        return xo



class TopDownLocal_FPNFuse(HybridBlock):
    def __init__(self, channels=64):
        super(TopDownLocal_FPNFuse, self).__init__()
        inter_channels = int(channels // 1)

        with self.name_scope():

            self.bn1 = nn.BatchNorm()
            self.bn2 = nn.BatchNorm()

            self.topdown_att = nn.HybridSequential(prefix='topdown_att')
            self.topdown_att.add(nn.Conv2D(inter_channels, kernel_size=1, strides=1, padding=0))
            self.topdown_att.add(nn.BatchNorm())
            self.topdown_att.add(nn.Activation('relu'))
            self.topdown_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.topdown_att.add(nn.BatchNorm())
            self.topdown_att.add(nn.Activation('sigmoid'))

    def hybrid_forward(self, F, x, residual):

        x = self.bn1(x)
        residual = self.bn2(residual)
        topdown_wei = self.topdown_att(x)

        xo = x + F.broadcast_mul(topdown_wei, residual)

        return xo


class GlobalSK_FPNFuse(HybridBlock):
    def __init__(self, channels=64, r=2):
        super(GlobalSK_FPNFuse, self).__init__()
        inter_channels = int(channels // r)

        with self.name_scope():

            self.bn1 = nn.BatchNorm()
            self.bn2 = nn.BatchNorm()

            self.global_att = nn.HybridSequential(prefix='global_att')
            self.global_att.add(nn.GlobalAvgPool2D())
            self.global_att.add(nn.Conv2D(inter_channels, kernel_size=1, strides=1, padding=0))
            self.global_att.add(nn.BatchNorm())
            self.global_att.add(nn.Activation('relu'))
            self.global_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.global_att.add(nn.BatchNorm())
            self.global_att.add(nn.Activation('sigmoid'))

    def hybrid_forward(self, F, x, residual):

        x = self.bn1(x)
        residual = self.bn2(residual)
        xa = x + residual

        wei = self.global_att(xa)

        xo = F.broadcast_mul(x, wei) + F.broadcast_mul(residual, 1-wei)

        return xo


class MutualSKLocal_FPNFuse(HybridBlock):
    def __init__(self, channels=64):
        super(MutualSKLocal_FPNFuse, self).__init__()
        inter_channels = int(channels // 2)

        with self.name_scope():

            self.bn1 = nn.BatchNorm()
            self.bn2 = nn.BatchNorm()

            self.topdown = nn.HybridSequential(prefix='topdown')
            self.topdown.add(nn.Conv2D(inter_channels, kernel_size=1, strides=1, padding=0))
            self.topdown.add(nn.BatchNorm())
            self.topdown.add(nn.Activation('relu'))
            self.topdown.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.topdown.add(nn.BatchNorm())
            self.topdown.add(nn.Activation('sigmoid'))

            self.bottomup = nn.HybridSequential(prefix='bottomup')
            self.bottomup.add(nn.Conv2D(inter_channels, kernel_size=1, strides=1, padding=0))
            self.bottomup.add(nn.BatchNorm())
            self.bottomup.add(nn.Activation('relu'))
            self.bottomup.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.bottomup.add(nn.BatchNorm())
            self.bottomup.add(nn.Activation('sigmoid'))

    def hybrid_forward(self, F, x, residual):

        x = self.bn1(x)
        residual = self.bn2(residual)
        xa = x + residual

        topdown_wei = self.topdown(xa)
        bottomup_wei = self.bottomup(xa)

        xo = F.broadcast_mul(x, topdown_wei) + F.broadcast_mul(residual, bottomup_wei)

        return xo


class MutualSKGlobal_FPNFuse(HybridBlock):
    def __init__(self, channels=64):
        super(MutualSKGlobal_FPNFuse, self).__init__()
        inter_channels = int(channels // 2)

        with self.name_scope():

            self.bn1 = nn.BatchNorm()
            self.bn2 = nn.BatchNorm()

            self.topdown = nn.HybridSequential(prefix='topdown')
            self.topdown.add(nn.GlobalAvgPool2D())
            self.topdown.add(nn.Conv2D(inter_channels, kernel_size=1, strides=1, padding=0))
            self.topdown.add(nn.BatchNorm())
            self.topdown.add(nn.Activation('relu'))
            self.topdown.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.topdown.add(nn.BatchNorm())
            self.topdown.add(nn.Activation('sigmoid'))

            self.bottomup = nn.HybridSequential(prefix='bottomup')
            self.bottomup.add(nn.GlobalAvgPool2D())
            self.bottomup.add(nn.Conv2D(inter_channels, kernel_size=1, strides=1, padding=0))
            self.bottomup.add(nn.BatchNorm())
            self.bottomup.add(nn.Activation('relu'))
            self.bottomup.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.bottomup.add(nn.BatchNorm())
            self.bottomup.add(nn.Activation('sigmoid'))

    def hybrid_forward(self, F, x, residual):

        x = self.bn1(x)
        residual = self.bn2(residual)
        xa = x + residual

        topdown_wei = self.topdown(xa)
        bottomup_wei = self.bottomup(xa)

        xo = F.broadcast_mul(x, topdown_wei) + F.broadcast_mul(residual, bottomup_wei)

        return xo



class LocalSK_FPNFuse(HybridBlock):
    def __init__(self, channels=64, r=2):
        super(LocalSK_FPNFuse, self).__init__()
        inter_channels = int(channels // r)

        with self.name_scope():

            self.bn1 = nn.BatchNorm()
            self.bn2 = nn.BatchNorm()

            self.global_att = nn.HybridSequential(prefix='global_att')
            self.global_att.add(nn.Conv2D(inter_channels, kernel_size=1, strides=1, padding=0))
            self.global_att.add(nn.BatchNorm())
            self.global_att.add(nn.Activation('relu'))
            self.global_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.global_att.add(nn.BatchNorm())
            self.global_att.add(nn.Activation('sigmoid'))

    def hybrid_forward(self, F, x, residual):

        x = self.bn1(x)
        residual = self.bn2(residual)
        xa = x + residual

        wei = self.global_att(xa)

        xo = F.broadcast_mul(x, wei) + F.broadcast_mul(residual, 1-wei)

        return xo



class ReMo_FPNFuse(HybridBlock):
    def __init__(self, channels=64, r=2):
        super(ReMo_FPNFuse, self).__init__()
        inter_channels = int(channels // r)

        with self.name_scope():

            self.global_att = nn.HybridSequential(prefix='global_att')
            self.global_att.add(nn.GlobalAvgPool2D())
            self.global_att.add(nn.Conv2D(inter_channels, kernel_size=1, strides=1, padding=0))
            self.global_att.add(nn.BatchNorm())
            self.global_att.add(nn.Activation('relu'))
            self.global_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.global_att.add(nn.BatchNorm())
            self.global_att.add(nn.Activation('sigmoid'))

            self.local_att = nn.HybridSequential(prefix='local_att')
            self.local_att.add(nn.Conv2D(inter_channels, kernel_size=1, strides=1, padding=0))
            self.local_att.add(nn.BatchNorm())
            self.local_att.add(nn.Activation('relu'))
            self.local_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.local_att.add(nn.BatchNorm())
            self.local_att.add(nn.Activation('sigmoid'))

    def hybrid_forward(self, F, x, residual):

        global_wei = self.global_att(x)
        local_wei = self.local_att(residual)

        wei = F.broadcast_add(global_wei, local_wei)
        xo = F.broadcast_mul(x + residual, wei)

        return xo



class LocalReMo_FPNFuse(HybridBlock):
    def __init__(self, channels=64, r=2):
        super(LocalReMo_FPNFuse, self).__init__()
        inter_channels = int(channels // r)

        with self.name_scope():

            self.global_att = nn.HybridSequential(prefix='global_att')
            self.global_att.add(nn.Conv2D(inter_channels, kernel_size=1, strides=1, padding=0))
            self.global_att.add(nn.BatchNorm())
            self.global_att.add(nn.Activation('relu'))
            self.global_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.global_att.add(nn.BatchNorm())
            self.global_att.add(nn.Activation('sigmoid'))

            self.local_att = nn.HybridSequential(prefix='local_att')
            self.local_att.add(nn.Conv2D(inter_channels, kernel_size=1, strides=1, padding=0))
            self.local_att.add(nn.BatchNorm())
            self.local_att.add(nn.Activation('relu'))
            self.local_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.local_att.add(nn.BatchNorm())
            self.local_att.add(nn.Activation('sigmoid'))

    def hybrid_forward(self, F, x, residual):

        global_wei = self.global_att(x)
        local_wei = self.local_att(residual)

        wei = F.broadcast_add(global_wei, local_wei)
        xo = F.broadcast_mul(x + residual, wei)

        return xo



class ResNetFCN(HybridBlock):
    def __init__(self, layers, channels, classes=1, norm_layer=BatchNorm, norm_kwargs=None,
                 **kwargs):
        super(ResNetFCN, self).__init__(**kwargs)

        self.layer_num = len(layers)
        with self.name_scope():

            stem_width = int(channels[0])
            self.stem = nn.HybridSequential(prefix='stem')
            self.stem.add(norm_layer(scale=False, center=False,
                                     **({} if norm_kwargs is None else norm_kwargs)))
            self.stem.add(nn.Conv2D(channels=stem_width*2, kernel_size=3, strides=1,
                                     padding=1, use_bias=False))
            self.stem.add(norm_layer(in_channels=stem_width*2))
            self.stem.add(nn.Activation('relu'))

            self.head = _FCNHead(in_channels=channels[-1], channels=classes)

            self.layer1 = self._make_layer(block=CIFARBasicBlockV1, layers=layers[0],
                                           channels=channels[1], stride=1, stage_index=1,
                                           in_channels=channels[1])

            self.layer2 = self._make_layer(block=CIFARBasicBlockV1, layers=layers[1],
                                           channels=channels[2], stride=2, stage_index=2,
                                           in_channels=channels[1])

            self.layer3 = self._make_layer(block=CIFARBasicBlockV1, layers=layers[2],
                                           channels=channels[3], stride=2, stage_index=3,
                                           in_channels=channels[2])

    def _make_layer(self, block, layers, channels, stride, stage_index, in_channels=0,
                    norm_layer=BatchNorm, norm_kwargs=None):
        layer = nn.HybridSequential(prefix='stage%d_'%stage_index)
        with layer.name_scope():
            downsample = (channels != in_channels) or (stride != 1)
            layer.add(block(channels, stride, downsample, in_channels=in_channels,
                            prefix='', norm_layer=norm_layer, norm_kwargs=norm_kwargs))
            for _ in range(layers-1):
                layer.add(block(channels, 1, False, in_channels=channels, prefix='',
                                norm_layer=norm_layer, norm_kwargs=norm_kwargs))
        return layer

    def hybrid_forward(self, F, x):

        _, _, orig_hei, orig_wid = x.shape
        x = self.stem(x)      # sub 2
        c1 = self.layer1(x)   # sub 2
        c2 = self.layer2(c1)  # sub 4
        c3 = self.layer3(c2)  # sub 8

        pred = self.head(c3)
        out = F.contrib.BilinearResize2D(pred, height=orig_hei, width=orig_wid)

        return out

    def evaluate(self, x):
        """evaluating network with inputs and targets"""
        return self.forward(x)


class ResNetFPN(HybridBlock):
    def __init__(self, layers, channels, classes=1, norm_layer=BatchNorm, norm_kwargs=None,
                 **kwargs):
        super(ResNetFCN, self).__init__(**kwargs)

        self.layer_num = len(layers)
        with self.name_scope():

            stem_width = int(channels[0])
            self.stem = nn.HybridSequential(prefix='stem')
            self.stem.add(norm_layer(scale=False, center=False,
                                     **({} if norm_kwargs is None else norm_kwargs)))
            self.stem.add(nn.Conv2D(channels=stem_width*2, kernel_size=3, strides=1,
                                     padding=1, use_bias=False))
            self.stem.add(norm_layer(in_channels=stem_width*2))
            self.stem.add(nn.Activation('relu'))

            self.head = _FCNHead(in_channels=channels[-1], channels=classes)

            self.layer1 = self._make_layer(block=CIFARBasicBlockV1, layers=layers[0],
                                           channels=channels[1], stride=1, stage_index=1,
                                           in_channels=channels[1])

            self.layer2 = self._make_layer(block=CIFARBasicBlockV1, layers=layers[1],
                                           channels=channels[2], stride=2, stage_index=2,
                                           in_channels=channels[1])

            self.layer3 = self._make_layer(block=CIFARBasicBlockV1, layers=layers[2],
                                           channels=channels[3], stride=2, stage_index=3,
                                           in_channels=channels[2])

    def _make_layer(self, block, layers, channels, stride, stage_index, in_channels=0,
                    norm_layer=BatchNorm, norm_kwargs=None):
        layer = nn.HybridSequential(prefix='stage%d_'%stage_index)
        with layer.name_scope():
            downsample = (channels != in_channels) or (stride != 1)
            layer.add(block(channels, stride, downsample, in_channels=in_channels,
                            prefix='', norm_layer=norm_layer, norm_kwargs=norm_kwargs))
            for _ in range(layers-1):
                layer.add(block(channels, 1, False, in_channels=channels, prefix='',
                                norm_layer=norm_layer, norm_kwargs=norm_kwargs))
        return layer

    def hybrid_forward(self, F, x):

        _, _, orig_hei, orig_wid = x.shape
        x = self.stem(x)      # sub 2
        c1 = self.layer1(x)   # sub 2
        c2 = self.layer2(c1)  # sub 4
        c3 = self.layer3(c2)  # sub 8

        pred = self.head(c3)
        out = F.contrib.BilinearResize2D(pred, height=orig_hei, width=orig_wid)

        return out

    def evaluate(self, x):
        """evaluating network with inputs and targets"""
        return self.forward(x)
