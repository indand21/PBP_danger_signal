"""Extract visible blue marker clusters from Sabet 2020 Figs 1-6.

Source PDF is local and gitignored. Only numerical observations and their
pixel coordinates are retained. Overlaid points are marker clusters, not
recovered individual animals. Dose/interval identities cannot be recovered.
Run from the repository root: python tools/digitize_sabet.py
"""
from pathlib import Path
import csv
import hashlib
import json

import fitz
import numpy as np
from PIL import Image
from scipy.ndimage import label, find_objects

ROOT = Path(__file__).resolve().parents[1]
SOURCE = next((ROOT / "docs/article_pdfs").glob("sabet*.pdf"))
OUT = ROOT / "data/external_validation/sabet2020_strain_points.csv"
QA = ROOT / "tmp/pd_source_review"
# Pixel coordinates in the original embedded figure images (not page PNGs).
# x0, x1, maximum x index; y at zero and y units per log10 decade.
CAL = {
    "AB1148": (2, 1, 349, 2953, 70, 568, 273),
    "AB1036": (2, 0, 169, 1475, 70, 379, 122.5),
    "AB1022": (3, 1, 295, 2950, 70, 722, 231.5),
    "AB1157": (3, 0, 167, 1474, 60, 310, 99.2),
    "AB1137": (4, 1, 167, 1573, 50, 470, 153),
    "AB1129": (4, 0, 170, 1476, 50, 352, 170),
}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    QA.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(SOURCE)
    records = []
    for strain, (page_no, image_no, x0, x1, xmax, y0, dy) in CAL.items():
        xref = doc[page_no].get_images()[image_no][0]
        blob = doc.extract_image(xref)
        import io
        im = np.array(Image.open(io.BytesIO(blob["image"])).convert("RGB"))
        rgb = im.astype(float)
        mask = ((rgb[:,:,2] > rgb[:,:,0]+18) & (rgb[:,:,2] > rgb[:,:,1]+12)
                & (rgb[:,:,0] < 110) & (rgb[:,:,2] < 210))
        labs, _ = label(mask)
        fig, ax = plt.subplots(figsize=(10,6))
        ax.imshow(im)
        for component, sl in enumerate(find_objects(labs), 1):
            if sl is None: continue
            yy, xx = np.where(labs[sl] == component)
            if len(xx) < 45: continue
            px, py = float(xx.mean()+sl[1].start), float(yy.mean()+sl[0].start)
            if px < x0-25 or px > x1+20: continue
            # Exclude the blue legend symbol, preserving observations at x=0.
            if px > .3*im.shape[1] and py < .20*im.shape[0]: continue
            width, height = sl[1].stop-sl[1].start, sl[0].stop-sl[0].start
            ix = max(0., (px-x0)/(x1-x0)*xmax)
            value = (y0-py)/dy
            quality = "overlaid_marker_cluster" if height > 1.35*width else "visible_marker"
            if ix < .5:
                ix = 0.
                quality = "zero_index_cluster_not_paired_to_regimen"
            records.append({"strain":strain, "index_pct":round(ix,2),
                            "delta_log10_cfu_thigh":round(value,2),
                            "pixel_x":round(px,1), "pixel_y":round(py,1),
                            "quality":quality})
            ax.plot(px,py,"o",mfc="none",mec="red",ms=8,mew=.7)
        ax.set_title(strain+" recovered marker clusters (red outlines)")
        ax.axis("off");fig.tight_layout()
        fig.savefig(QA/f"sabet_{strain}_readings.png",dpi=140);plt.close(fig)
    records.sort(key=lambda r:(list(CAL).index(r["strain"]), r["index_pct"],-r["delta_log10_cfu_thigh"]))
    with OUT.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    meta = {"source_pdf_sha256":hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "calibration":CAL,"n_marker_clusters":len(records),
            "control_timing": {"treatment_start_h_postinfection":2,
                               "untreated_sacrifice_h_postinfection":12,
                               "elapsed_control_growth_h":10,
                               "reported_delta_log10_range":[1.16,2.08]},
            "caveats": ["Visible marker clusters, not recovered replicate-level animal data",
                        "No dose or dosing-interval identity is encoded by the blue markers",
                        "Zero-index points cannot all be assigned to untreated animals with certainty",
                        "Table 3 gives AB1129 two-log 46.6; Fig 6 prints 46.9. Table 3 retained.",
                        "Table 3 gives AB1137 one-log 30.1; Fig 5 prints 30.0. Table 3 retained."]}
    OUT.with_suffix(".provenance.json").write_text(json.dumps(meta,indent=2)+"\n",encoding="utf-8")
    print("Saved",len(records),"marker clusters")


if __name__=="__main__":
    main()
