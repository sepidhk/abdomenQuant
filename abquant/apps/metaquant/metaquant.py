import argparse
import os
import json
from abquant.dicomseries import DicomSeries
from abquant.apps.l3_detection.predict_l3_slice import main as predict_l3_slice
from abquant.apps.wcirc.utils import get_waist_circumference


parser = argparse.ArgumentParser(description='Quantify Metabolic Status from Axial Scan')
parser.add_argument('--dicom-dir', type=str, help='Directory containing the dicom files')
parser.add_argument('--slice-model-path', type=str, help='Path to the L3 detection model')
parser.add_argument('--slice-model-weights', type=str, help='Path to the L3 detection model weights')
parser.add_argument('--output-dir', type=str, help='Directory to save the output')
args = parser.parse_args()


def main(args):
    # Load the dicom series
    dicom_series = DicomSeries(args.dicom_dir, filepattern='*', window_center=40, window_width=1200, read_images=True)
    output_dir = f'{args.output_dir}/MRN{dicom_series.mrn}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    l3_slice, l3_info = predict_l3_slice(args, dicom_series, output_dir)
    waist_circ = get_waist_circumference(dicom_series.pixel_array, l3_slice, dicom_series.spacing)
    l3_info['waist_circ'] = waist_circ
    json.dump(l3_info, open(f'{output_dir}/{dicom_series.mrn}_{dicom_series.accession}_{dicom_series.cut}_l3_info.json', 'w'))
    return l3_info


if __name__ == '__main__':
    main(args)
