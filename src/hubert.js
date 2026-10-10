export class Hubert {
  /** session: ONNX export of HubertModel (+final_proj for v1). input [1,N] float32 -> output [1,T,C]. */
  constructor(ort, session) { this.ort = ort; this.session = session; }
  async run(audio16k) {
    const input = new this.ort.Tensor("float32", audio16k, [1, audio16k.length]);
    const out = await this.session.run({ [this.session.inputNames[0]]: input });
    const t = out[this.session.outputNames[0]];
    return { data: t.data, T: t.dims[1], C: t.dims[2] };
  }
}
