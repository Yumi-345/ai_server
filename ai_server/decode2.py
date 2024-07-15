import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GObject, GLib

Gst.init(None)

pipeline = Gst.Pipeline.new("test")

#rtspsrc的srcpad是随机衬垫，这里使用回调函数来连接衬垫。
def on_pad_added( src, pad, des):
    print(2)
    vpad = des.get_static_pad("sink")
    pad.link(vpad)

def create_source_bin(index,uri):

    bin_name="source-bin-%02d" %index
    nbin=Gst.Bin.new(bin_name)
    pipeline.add(nbin)

    src = Gst.ElementFactory.make("rtspsrc", "src-"+str(index))
    src.set_property("location", uri)

    Gst.Bin.add(nbin,src)

    queuev1 = Gst.ElementFactory.make("queue2", "queue-"+str(index))
    src.connect("pad-added", on_pad_added, queuev1)
    Gst.Bin.add(nbin,queuev1)

    depay = Gst.ElementFactory.make("rtph265depay", "depay-"+str(index))
    Gst.Bin.add(nbin,depay)

    parse = Gst.ElementFactory.make("h265parse", "parse-"+str(index))
    Gst.Bin.add(nbin,parse)

    decode = Gst.ElementFactory.make("nvv4l2decoder", "decoder-"+str(index))
    Gst.Bin.add(nbin,decode)
    decode.set_property("enable-max-performance", True)
    decode.set_property("drop-frame-interval", 5)      # important!!!!!!!!!!!!!!!!!!!!!!!!!!!
    decode.set_property("num-extra-surfaces", 0)

    nbin.add_pad(Gst.GhostPad.new_no_target("src",Gst.PadDirection.SRC))

    queuev1.link(depay)
    depay.link(parse)
    parse.link(decode)



    decoder_src_pad = decode.get_static_pad("src")
    caps=decoder_src_pad.get_current_caps()
    if not caps:
        caps = decoder_src_pad.query_caps()
    gststruct=caps.get_structure(0)
    gstname=gststruct.get_name()

    features=caps.get_features(0)

    # Need to check if the pad created by the decodebin is for video and not
    # audio.
    print("gstname=",gstname)
    print("features=",features)
    if features.contains("memory:NVMM"):
        bin_ghost_pad=nbin.get_static_pad("src")
        bin_ghost_pad.set_target(decoder_src_pad)
        sinkpad = testdrmsink.get_static_pad("sink")


        bin_ghost_pad.link(sinkpad)
    else:
        print("not contains")

    # fakesink = Gst.ElementFactory.make("nvdrmvideosink", "testdrmsink")
    # Gst.Bin.add(nbin,fakesink)
    
    # decode.link(testdrmsink)

    # srcpad = nbin.get_static_pad("nbinsrcpad")
    # sinkpad = fakesink.get_static_pad("sink")
    # decodesrcpad.link(sinkpad)

    return nbin

testdrmsink = Gst.ElementFactory.make("nvdrmvideosink", "testdrmsink")
pipeline.add(testdrmsink)

nbin = create_source_bin(0, "rtsp://192.168.31.191:551/2160")

pipeline.set_state(Gst.State.PLAYING)

mainloop = GLib.MainLoop()
mainloop.run()