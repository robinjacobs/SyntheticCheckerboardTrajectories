from synthgen.constants import ATRACSYS_SERIAL


template_calibration_dict = {
    "serial": "",
    "type": "",
    "modality": "",
    "recording": "",
    "markerGeometries": "",
    "ref2DevClock": "345999960215",
    "intrinsics": {
        "type_id": "opencv-matrix",
        "rows": 3,
        "cols": 3,
        "dt": "d",
        "data": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    },
    "distCoeffs": {
        "type_id": "opencv-matrix",
        "rows": 0,
        "cols": 0,
        "dt": "u",
        "data": [],
    },
    "cam2devRef": {
        "type_id": "opencv-matrix",
        "rows": 4,
        "cols": 4,
        "dt": "d",
        "data": [
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
    },
    "extrinsicsCalibrated": "0",
    "devRef2world": {
        "type_id": "opencv-matrix",
        "rows": 4,
        "cols": 4,
        "dt": "d",
        "data": [
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
    },
    "extrinsicsError": "-1.000000",
    "extrinsicsInlierRatio": "-1.000000",
}


template_pre_calibration_dict = {
    "serial": str(ATRACSYS_SERIAL),
    "type": "FusionTrack 500",
    "modality": "",
    "recording": "",
    "markerGeometries": "model_to_marker.json",
    "ref2DevClock": "-1661852848349079",
    "intrinsics": {
        "type_id": "opencv-matrix",
        "rows": 3,
        "cols": 3,
        "dt": "d",
        "data": [
            4.6606543361948221e-310,
            0.0,
            4.6606543872614472e-310,
            9.8813129168249309e-324,
            0.0,
            6.5216665251044544e-322,
            6.9532219467126241e-310,
            0.0,
            1.0573004821002676e-321,
        ],
    },
    "distCoeffs": {
        "type_id": "opencv-matrix",
        "rows": 0,
        "cols": 0,
        "dt": "u",
        "data": [],
    },
    "cam2devRef": {
        "type_id": "opencv-matrix",
        "rows": 4,
        "cols": 4,
        "dt": "d",
        "data": [
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
    },
    "extrinsicsCalibrated": "1",
    "devRef2world": {
        "type_id": "opencv-matrix",
        "rows": 4,
        "cols": 4,
        "dt": "d",
        "data": [
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
    },
    "extrinsicsError": "0.000000",
    "extrinsicsInlierRatio": "-1.000000",
}
