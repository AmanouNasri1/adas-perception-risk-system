"""
Build the V7 technical report (docs/technical_report.pdf), structured per the project
plan (PDF section 15.3). Reproducible:

    python scripts/build_report.py

Figures come from outputs/figures/; numbers mirror the README Results section.
"""
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, HRFlowable, KeepTogether,
)

FIG = Path("outputs/figures")
NAVY = colors.HexColor("#0E1B2C")
AMBER = colors.HexColor("#C8860A")
TEAL = colors.HexColor("#1C7293")
GREY = colors.HexColor("#5A6B7B")
LIGHT = colors.HexColor("#EEF3F8")

PAGE_W, PAGE_H = A4
MARGIN = 2.0 * cm
CONTENT_W = PAGE_W - 2 * MARGIN

styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontName="Helvetica-Bold",
                    fontSize=15, textColor=NAVY, spaceBefore=16, spaceAfter=7, leading=18)
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontName="Helvetica-Bold",
                    fontSize=12, textColor=TEAL, spaceBefore=10, spaceAfter=4, leading=15)
BODY = ParagraphStyle("Body", parent=styles["BodyText"], fontName="Helvetica",
                      fontSize=10, leading=14.5, alignment=TA_JUSTIFY, spaceAfter=6)
ABSTRACT = ParagraphStyle("Abstract", parent=BODY, leftIndent=0.5 * cm,
                          rightIndent=0.5 * cm, fontSize=9.5, leading=14, textColor=colors.HexColor("#222222"))
CAP = ParagraphStyle("Cap", parent=styles["Normal"], fontName="Helvetica-Oblique",
                     fontSize=8.5, textColor=GREY, alignment=TA_CENTER, spaceBefore=3, spaceAfter=10)
BULLET = ParagraphStyle("Bullet", parent=BODY, leftIndent=0.6 * cm, bulletIndent=0.2 * cm,
                        spaceAfter=3)


def fig(name, max_w=CONTENT_W, max_h=8.5 * cm, caption=None):
    p = FIG / name
    iw, ih = PILImage.open(p).size
    ar = iw / ih
    w = max_w
    h = w / ar
    if h > max_h:
        h = max_h
        w = h * ar
    img = Image(str(p), width=w, height=h)
    img.hAlign = "CENTER"
    flow = [img]
    if caption:
        flow.append(Paragraph(caption, CAP))
    return KeepTogether(flow)


def table(data, col_w, header=True, font=8.8):
    t = Table(data, colWidths=col_w, hAlign="LEFT")
    st = [
        ("FONT", (0, 0), (-1, -1), "Helvetica", font),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#1a1a1a")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#CFD8E0")),
    ]
    if header:
        st += [
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", font),
            ("TOPPADDING", (0, 0), (-1, 0), 5),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 5),
        ]
        st += [("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT])]
    t.setStyle(TableStyle(st))
    return t


def b(text):
    return Paragraph(text, BULLET, bulletText="•")


def _header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(GREY)
    canvas.drawString(MARGIN, 1.1 * cm, "Real-Time ADAS Perception and Risk Warning System")
    canvas.drawRightString(PAGE_W - MARGIN, 1.1 * cm, f"Page {doc.page}")
    canvas.setStrokeColor(colors.HexColor("#CFD8E0"))
    canvas.line(MARGIN, 1.35 * cm, PAGE_W - MARGIN, 1.35 * cm)
    canvas.restoreState()


def build():
    story = []
    A = story.append

    # ---- Title block ----
    A(Spacer(1, 2.2 * cm))
    A(Paragraph("Real-Time ADAS Perception<br/>and Risk Warning System",
                ParagraphStyle("T", parent=styles["Title"], fontName="Helvetica-Bold",
                               fontSize=24, textColor=NAVY, leading=29, alignment=TA_CENTER)))
    A(Spacer(1, 0.3 * cm))
    A(Paragraph("Monocular detection, tracking, lane segmentation, distance, "
                "time-to-collision, and tiered risk warnings on driving video",
                ParagraphStyle("ST", parent=BODY, fontSize=12, alignment=TA_CENTER,
                               textColor=TEAL, leading=16)))
    A(Spacer(1, 0.6 * cm))
    A(HRFlowable(width="40%", thickness=1.2, color=AMBER, hAlign="CENTER"))
    A(Spacer(1, 0.5 * cm))
    A(Paragraph("Technical Report", ParagraphStyle("rt", parent=BODY, fontSize=13,
                alignment=TA_CENTER, textColor=NAVY, fontName="Helvetica-Bold")))
    A(Paragraph("Amanou Allah Nasri", ParagraphStyle("au", parent=BODY, fontSize=11,
                alignment=TA_CENTER, textColor=colors.black)))
    A(Paragraph("github.com/AmanouNasri1/adas-perception-risk-system",
                ParagraphStyle("repo", parent=BODY, fontSize=9.5, alignment=TA_CENTER, textColor=TEAL)))
    A(Spacer(1, 1.0 * cm))
    A(fig("v6_lane_demo.png", max_w=14 * cm, max_h=8 * cm,
          caption="The full pipeline: drivable-area front zone (green), lane lines (red), "
                  "tracked vehicles with distance, and HUD warnings."))
    A(PageBreak())

    # ---- Abstract ----
    A(Paragraph("Abstract", H1))
    A(Paragraph(
        "This report describes a real-time Advanced Driver-Assistance System (ADAS) "
        "perception pipeline that operates on a single forward-facing camera. The system "
        "chains YOLO11 object detection, ByteTrack multi-object tracking, YOLOPv2 lane/road-"
        "area segmentation, monocular distance estimation, time-to-collision (TTC) reasoning, "
        "and a tiered warning engine, producing an annotated video together with frame-level "
        "CSV logs and runtime metrics. The detector, fine-tuned on KITTI, reaches "
        "mAP@0.5&nbsp;=&nbsp;0.922. The pipeline runs at 34&nbsp;FPS on an RTX&nbsp;3050 "
        "laptop GPU, or 15.6&nbsp;FPS with deep lane segmentation enabled. Two design "
        "decisions are validated by A/B measurement: an in-path gate reduces time-to-collision "
        "false positives by 75&nbsp;%, and replacing the fixed front-zone trapezoid with the "
        "segmented ego-drivable area reduces front-object false positives by 67&nbsp;%. The "
        "project is built as clean, measured, reproducible engineering and is intended as the "
        "foundation for a bachelor thesis.", ABSTRACT))

    # ---- 1. Introduction ----
    A(Paragraph("1&nbsp;&nbsp;Introduction", H1))
    A(Paragraph(
        "ADAS reduce collisions by perceiving the road scene and warning the driver about "
        "imminent risk. While production systems fuse radar, lidar and cameras, a large part "
        "of the perceptual signal can be recovered from a single camera, which makes monocular "
        "ADAS an attractive low-cost setting and a faithful proxy for the perception stack of "
        "autonomous vehicles. The goal of this project is not a flashy demo but an honest, "
        "measured engineering pipeline: each stage is testable from the command line, every "
        "performance claim is backed by numbers, and failure cases are documented rather than "
        "hidden.", BODY))
    A(Paragraph(
        "The system was developed in seven incremental phases (V1&ndash;V7): a baseline "
        "detection/tracking/warning demo, an evaluation pass, KITTI fine-tuning, distance "
        "estimation, time-to-collision, lane/road-area awareness, and this publication "
        "package. Each phase met explicit acceptance criteria before the next began.", BODY))

    # ---- 2. System architecture ----
    A(Paragraph("2&nbsp;&nbsp;System architecture", H1))
    A(Paragraph(
        "The system is one pipeline, not a set of disconnected scripts. Reusable logic lives "
        "in a Python package (<font face='Courier'>adas_perception</font>) with submodules for "
        "detection, tracking, risk, visualization and utilities; thin CLI scripts import from "
        "it. The processing chain is:", BODY))
    A(Paragraph(
        "<b>frame loader &rarr; YOLO11 detector &rarr; ByteTrack tracker &rarr; YOLOPv2 "
        "lane/road segmentation (dynamic front zone) &rarr; risk-zone analyzer &rarr; "
        "monocular distance estimator &rarr; time-to-collision &rarr; tiered warning engine "
        "&rarr; annotated video + CSV logs + metrics</b>.", BODY))
    A(Paragraph(
        "Configuration is externalised: thresholds, zone geometry, model paths and class "
        "lists live in <font face='Courier'>configs/*.yaml</font>, and CLI flags override "
        "them. Generated artifacts (videos, logs, figures) are written under "
        "<font face='Courier'>outputs/</font> and are git-ignored; datasets and weights are "
        "kept on external storage and never committed.", BODY))

    # ---- 3. Implementation ----
    A(Paragraph("3&nbsp;&nbsp;Implementation", H1))
    A(Paragraph(
        "The stack is Python with Ultralytics YOLO, PyTorch and OpenCV. Development is on "
        "Windows with an RTX&nbsp;3050; training runs in Google Colab. Key components:", BODY))
    A(b("<b>Detection</b> &mdash; YOLO11s, fine-tuned on KITTI; configurable confidence/IoU/"
        "image size in <font face='Courier'>configs/detector.yaml</font>."))
    A(b("<b>Tracking</b> &mdash; ByteTrack by default (BoT-SORT optional), giving persistent "
        "track IDs written to <font face='Courier'>tracking_results.csv</font>."))
    A(b("<b>Lane/road area</b> &mdash; YOLOPv2 (TorchScript) drivable-area and lane-line "
        "segmentation; the EMA-smoothed drivable region becomes the dynamic front zone, with "
        "automatic fallback to a static trapezoid when confidence is low."))
    A(b("<b>Distance</b> &mdash; pinhole model <font face='Courier'>D = f_y &middot; H_real / "
        "h_bbox</font>, using a heuristic focal length or a KITTI P2 calibration."))
    A(b("<b>Time-to-collision</b> &mdash; per-track distance ring buffer with a least-squares "
        "closing-speed fit; <font face='Courier'>TTC = distance / approach_speed</font>."))
    A(b("<b>Warning engine</b> &mdash; tiered warnings (Section&nbsp;5) drawn on the video "
        "and logged per row in <font face='Courier'>warnings.csv</font>."))

    # ---- 4. Experiments ----
    A(Paragraph("4&nbsp;&nbsp;Experiments", H1))
    A(Paragraph(
        "All runtime figures use the same clip: a 1280&times;720, 30&nbsp;fps dashcam video of "
        "4967 frames, on an RTX&nbsp;3050 laptop GPU. Detection accuracy is measured on the "
        "KITTI validation split after fine-tuning. The detector was trained for 50 epochs on a "
        "Colab T4 (116&nbsp;minutes).", BODY))
    A(Paragraph("4.1&nbsp;&nbsp;Detector fine-tuning (KITTI)", H2))
    A(table([
        ["Metric", "Value"],
        ["mAP@0.5", "0.922"],
        ["mAP@0.5:0.95", "0.700"],
        ["Precision", "0.924"],
        ["Recall", "0.855"],
        ["Training time", "116 min (Colab T4, 50 epochs)"],
    ], [6 * cm, 9 * cm]))
    A(Spacer(1, 0.3 * cm))
    A(fig("kitti_training_curve.png", max_h=6 * cm,
          caption="Figure 1. KITTI fine-tuning curve (loss and mAP over epochs)."))
    A(Paragraph("4.2&nbsp;&nbsp;Tracker comparison", H2))
    A(Paragraph("ByteTrack is the default for speed; BoT-SORT yields more stable IDs.", BODY))
    A(table([
        ["Metric", "ByteTrack", "BoT-SORT"],
        ["Unique track IDs (fewer = stable)", "126", "115"],
        ["Avg track length (frames)", "27.9", "31.8"],
        ["Median track length (frames)", "6", "9"],
        ["Tracks ≥ 10 frames", "51", "53"],
    ], [8 * cm, 3.5 * cm, 3.5 * cm]))

    # ---- 5. Results ----
    A(Paragraph("5&nbsp;&nbsp;Results", H1))
    A(Paragraph("5.1&nbsp;&nbsp;Runtime", H2))
    A(table([
        ["Configuration", "Avg FPS", "Notes"],
        ["Baseline (no lane segmentation)", "34.0", "YOLO + tracking + zones + distance + TTC"],
        ["With V6 lane segmentation", "15.6", "+ YOLOPv2 ~24.9 ms/frame (fp16)"],
    ], [7.5 * cm, 2.5 * cm, 5 * cm]))
    A(Spacer(1, 0.25 * cm))
    A(fig("fps_plot.png", max_h=5.5 * cm, caption="Figure 2. Per-frame processing FPS over the clip."))
    A(Paragraph("5.2&nbsp;&nbsp;Warning tiers", H2))
    A(Paragraph("Warnings are prioritised by severity; the highest-priority active warning sets "
                "the bounding-box colour and headline alert.", BODY))
    A(table([
        ["Priority", "Warning", "Trigger"],
        ["5", "TTC WARNING", "Imminent collision, TTC < 3 s (in-path objects)"],
        ["4", "PEDESTRIAN / CYCLIST RISK", "Vulnerable road user in the pedestrian zone"],
        ["3", "VEHICLE TOO CLOSE", "Large bbox near the bottom of the frame"],
        ["1", "FRONT OBJECT", "Object inside the ego drivable lane"],
    ], [2 * cm, 5.5 * cm, 7.5 * cm]))
    A(Spacer(1, 0.25 * cm))
    A(fig("v5_ttc_frame.png", max_h=6.5 * cm,
          caption="Figure 3. HUD showing per-track distance, time-to-collision, and active warnings."))
    A(Paragraph("5.3&nbsp;&nbsp;Lane-aware front zone (V6 A/B)", H2))
    A(Paragraph(
        "Replacing the static trapezoid with the YOLOPv2 ego-drivable area restricts FRONT "
        "OBJECT to vehicles actually in the ego lane. On the test clip the lane was detected "
        "in 98.0&nbsp;% of frames (the remainder fall back to the static zone). The warning "
        "breakdown below isolates the effect: only FRONT OBJECT (and the TTC tier gated on it) "
        "changes; the depth- and pedestrian-zone-based tiers are untouched.", BODY))
    A(table([
        ["Warning type", "Static zone", "V6 lane zone", "Delta"],
        ["FRONT OBJECT", "3637", "1215", "-67%"],
        ["TTC WARNING", "1559", "1446", "-7%"],
        ["VEHICLE TOO CLOSE", "2329", "2329", "0"],
        ["PEDESTRIAN RISK", "116", "116", "0"],
        ["CYCLIST RISK", "5", "5", "0"],
    ], [6 * cm, 3 * cm, 3 * cm, 3 * cm]))
    A(Spacer(1, 0.25 * cm))
    A(fig("v6_static_vs_lane_4200.png", max_h=8.5 * cm,
          caption="Figure 4. Static front zone (top) vs V6 lane-aware drivable zone (bottom) "
                  "on the same frame. Adjacent-lane vehicles are no longer flagged."))
    A(Paragraph("5.4&nbsp;&nbsp;Time-to-collision gating", H2))
    A(Paragraph(
        "Naive monocular TTC fires for every object whose bounding box grows, including parked "
        "and roadside vehicles the ego merely drives past. Gating TTC so it only escalates "
        "objects already raising an in-path zone warning removes these false positives:", BODY))
    A(table([
        ["TTC configuration", "TTC warnings"],
        ["Ungated (every tracked object)", "6181"],
        ["In-path gated (used)", "1559"],
        ["False positives removed", "4622  (75%)"],
    ], [9 * cm, 6 * cm]))

    # ---- 6. Failure cases ----
    A(Paragraph("6&nbsp;&nbsp;Failure cases", H1))
    A(Paragraph(
        "Documented failure modes, with the phase that addresses each:", BODY))
    A(b("<b>Front-zone over-firing (V1)</b> &mdash; the static trapezoid flagged adjacent-lane "
        "and roadside vehicles. Largely addressed in V6 by the lane-aware zone (&minus;67%); "
        "residual cases occur at intersections, where the system falls back to the static zone."))
    A(b("<b>Distant-pedestrian warnings</b> &mdash; image-space zones lack a depth cue, so a "
        "pedestrian at 20&nbsp;m triggers the same warning as one at 3&nbsp;m. Mitigation: a "
        "bbox-size or distance gate on the pedestrian zone (future work)."))
    A(b("<b>Stationary large vehicles</b> &mdash; the bbox-height heuristic flags a stationary "
        "bus/truck that fills the frame. TTC distinguishes stationary from approaching targets, "
        "since a stationary vehicle has near-zero closing speed."))
    A(Spacer(1, 0.2 * cm))
    A(fig("v6_fallback_intersection.png", max_h=6 * cm,
          caption="Figure 5. Intersection fallback: with no clear ego lane, the drivable-area "
                  "zone is low-confidence and the static front zone is used."))

    # ---- 7. Limitations ----
    A(Paragraph("7&nbsp;&nbsp;Limitations", H1))
    A(b("<b>Monocular depth is approximate.</b> Bounding-box-height distance is biased by "
        "object-size priors and camera pitch; it is labelled approximate and is not used for "
        "safety-critical decisions."))
    A(b("<b>Lane segmentation degrades at intersections</b> and in adverse lighting/weather; "
        "the static fallback is a heuristic, not a solution."))
    A(b("<b>No ground-truth validation yet</b> for distance, in-path classification, or TTC; "
        "current evidence is internal A/B and qualitative inspection."))
    A(b("<b>Single test clip and single GPU</b> for runtime; broader hardware and footage "
        "diversity remain future work."))

    # ---- 8. Future work ----
    A(Paragraph("8&nbsp;&nbsp;Future work", H1))
    A(b("Ground-truth evaluation on KITTI tracking: distance error by range/class, FRONT "
        "OBJECT precision/recall vs in-path labels, and TTC false-positive/negative rates."))
    A(b("Learned monocular depth as a comparison baseline to the geometric estimator."))
    A(b("Deployment study: fp16/ONNX export and an accuracy-vs-FPS operating point."))
    A(b("Extension toward a bachelor thesis (see <i>thesis_proposal.md</i>), and optionally "
        "ROS&nbsp;2 / CARLA integration."))

    # ---- References ----
    A(Paragraph("References", H1))
    refs = [
        "Ultralytics YOLO. https://docs.ultralytics.com/",
        "Zhang et al. ByteTrack: Multi-Object Tracking by Associating Every Detection Box. ECCV 2022.",
        "Han et al. YOLOPv2: Better, Faster, Stronger for Panoptic Driving Perception. 2022. "
        "https://github.com/CAIC-AD/YOLOPv2",
        "Geiger et al. The KITTI Vision Benchmark Suite. https://www.cvlibs.net/datasets/kitti/",
        "PyTorch. Get Started Locally. https://pytorch.org/get-started/locally/",
    ]
    for i, r in enumerate(refs, 1):
        A(Paragraph(f"[{i}]&nbsp;&nbsp;{r}", ParagraphStyle("ref", parent=BODY, fontSize=9,
                    leading=13, spaceAfter=4, alignment=TA_JUSTIFY)))

    out = Path("docs/technical_report.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(out), pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN, topMargin=2.0 * cm, bottomMargin=1.8 * cm,
        title="Real-Time ADAS Perception and Risk Warning System — Technical Report",
        author="Amanou Allah Nasri",
    )
    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
    print(f"wrote {out}")


if __name__ == "__main__":
    build()
