import onnx
m = onnx.load("/home/simonli/ObjectDetection/yolo/weights/t2_static.onnx")
ops = {n.op_type for n in m.graph.node}
print("QuantizeLinear" in ops, "DequantizeLinear" in ops)
print([n.op_type for n in m.graph.node if n.op_type in ("QuantizeLinear","DequantizeLinear")][:10])
