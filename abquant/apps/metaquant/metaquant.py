import argparse
import os
import json
import matplotlib.pyplot as plt

from abquant.dicomseries import DicomSeries
from abquant.apps.slice_detection.predict_slice import main as predict_slice
from abquant.apps.wcirc.utils import get_waist_circumference, plot_contours


parser = argparse.ArgumentParser(description='Quantify Metabolic Status from Axial Scan')
parser.add_argument('--dicom-dir', type=str, help='Directory containing the dicom files')
parser.add_argument('--slice-model-path', default='abquant/models/CNNLine.json', type=str, help='Path to the slice detection model')
parser.add_argument('--l1-weights', type=str, default='abquant/models/l1_transfer_weights.h5', help='Path to the l1 model weights')
parser.add_argument('--l3-weights', type=str, default='abquant/models/CNNLine.h5', help='Path to the L3 detection model weights')
parser.add_argument('--l5-weights', type=str, default='abquant/models/l5_transfer_weights.h5', help='Path to the l5 model weights')
parser.add_argument('--output-dir', type=str, help='Directory to save the output')
parser.add_argument('--plot-outputs', type=bool, help='Plot the outputs', default=True)
args = parser.parse_args()


def main(args):
    # Load the dicom series
    dicom_series = DicomSeries(args.dicom_dir, filepattern='*', window_center=40, window_width=1200, read_images=True)
    output_dir = f'{args.output_dir}/MRN{dicom_series.mrn}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    series_info = dicom_series.series_info
    for vertebra in ['l1', 'l3', 'l5']:
        slice_location, slice_info = predict_slice(args, vertebra, dicom_series, output_dir)
        print(slice_info)
        for key, value in slice_info.items():
            series_info[key] = value
    if args.plot_outputs:
        fig = plt.imshow(dicom_series.frontal)
        for vertebra in ['l1', 'l3', 'l5']:
            plt.axhline(series_info[vertebra], color='y')
        plt.savefig(f'{output_dir}/{dicom_series.mrn}_{dicom_series.accession}_{dicom_series.cut}_slice_overlay.png')
    l3_body, waist_circ = get_waist_circumference(dicom_series.pixel_array, series_info['l3'], dicom_series.spacing)
    series_info['l3_waist_circ'] = waist_circ

    json.dump(series_info, open(f'{output_dir}/{dicom_series.mrn}_{dicom_series.accession}_{dicom_series.cut}_info.json', 'w'))
    if args.plot_outputs:
        outfile = f'{output_dir}/{dicom_series.mrn}_{dicom_series.accession}_{dicom_series.cut}_l3_waist_contour.png'
        plot_contours(dicom_series.pixel_array[series_info['l3']], l3_body, outfile)
    return series_info


if __name__ == '__main__':
    main(args)
