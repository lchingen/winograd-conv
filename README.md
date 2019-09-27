# Winograd Convolution

<p align='justify'>
This is a repository containing the source code for Winograd Convolution F(2,3).
It is tested for single patch Winograd convolution, as well as the streaming for
multiple patches that make up a larger input activation (IA).
</p>

## Sample Dataflow

```python

    # IA and W is pre-transformed (Winograd)
    for k in K:
        bias = B[k]
        for h in H:
            for w in W:
                for c in C:
                    ia_t = IA[h, w, c]
                    w_t  = W[k, c, ...]
                    oa_t = F23(ia,w)
                    oa = inv_wino(oa_t)
                    OA[c_idx, h_idx ,w_idx] += oa

                OA[c_idx, ...] += bias
                writeback_to_mm(OA)

```

<p align='justify'>
This dataflow suggests that the convolution partial sum be reduced fully in the
channel direction locally before triggering writeback to main memory. This is
due to potential limitations of limited on-chip memory storage.
</p>
