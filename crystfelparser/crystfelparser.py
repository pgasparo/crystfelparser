"""Console script for crystfelparser."""

from collections import defaultdict
import numpy as np
import argparse
import sys
import re
import subprocess

from crystfelparser.utils import save_dict_to_hdf5


def stream_to_dictionary(streamfile):
    """
    Function for parsing a indexamajig output stream

    Args:
      h5file: stream file to parse

    Returns:
      A dictionary
    """
    series = defaultdict(dict)
    series = dict()

    def loop_over_next_N_lines(file, n_lines):

        for cnt_tmp in range(n_lines):
            line = file.readline()

        return line

    with open(streamfile, "r") as text_file:
        # for ln,line in enumerate(text_file):
        ln = -1
        while True:
            ln += 1
            line = text_file.readline()
            # if any(x in ["Begin","chunk"] for x in line.split()):
            if "Begin chunk" in line:
                # create a temporary dictionary to store the output for a frame
                # tmpframe = defaultdict(int)
                tmpframe = dict()

                # loop over the next 3 lines to get the index of the image
                # line 2 and 3 are where it is stored the image number
                line = loop_over_next_N_lines(text_file, 3)
                ln += 3
                # save the image index and save it as zero-based
                im_num = np.int(line.split()[-1]) - 1
                tmpframe["Image serial number"] = im_num

                # loop over the next 2 lines to see if the indexer worked
                line = loop_over_next_N_lines(text_file, 2)
                ln += 2
                # save who indexed the image
                indexer_tmp = line.split()[-1]
                # if indexed, there is an additional line here
                tmpframe["indexed_by"] = indexer_tmp

                ##### Get the STRONG REFLEXTIONS from the spotfinder #####
                keyw=""
                while keyw != "num_peaks":
                    # loop over the next 5/6 lines to get the number of reflctions
                    line = loop_over_next_N_lines(text_file, 1)
                    ln += 1
                    try:
                        keyw=line.split()[0]
                    except:
                        keyw=""
                        
                # get the number of peaks
                num_peaks = np.int(line.split()[-1])
                tmpframe["num_peaks"] = num_peaks

                # get the resolution
                line = text_file.readline()
                ln += 1
                tmpframe["peak_resolution [A]"] = np.float(line.split()[-2])
                tmpframe["peak_resolution [nm^-1]"] = np.float(line.split()[2])

                if num_peaks > 0:
                    # skip the first 2 lines
                    for tmpc in range(2):
                        text_file.readline()
                        ln += 1

                    # get the spots
                    # fs/px, ss/px, (1/d)/nm^-1, Intensity
                    # with
                    # dim1 = ss, dim2 = fs
                    tmpframe["peaks"] = np.asarray(
                        [text_file.readline().split()[:4] for tmpc in range(num_peaks)]
                    ).astype(np.float)

                ##### Get the PREDICTIONS after indexing #####

                if tmpframe["indexed_by"] != "none":
                    # skip the first 2 header lines
                    for tmpc in range(2):
                        text_file.readline()
                        ln += 1
                    # Get the unit cell -- as cell lengths and angles
                    line = text_file.readline().split()
                    tmpframe["Cell parameters"] = np.hstack(
                        [line[2:5], line[6:9]]
                    ).astype(np.float)

                    # Get the reciprocal unit cell as a 3x3 matrix
                    reciprocal_cell = []
                    for tmpc in range(3):
                        reciprocal_cell.append(text_file.readline().split()[2:5])
                        ln += 1
                    tmpframe["reciprocal_cell_matrix"] = np.asarray(
                        reciprocal_cell
                    ).astype(np.float)

                    # Save the lattice type
                    tmpframe["lattice_type"] = text_file.readline().split()[-1]
                    ln += 1

                    # loop over the next 5 lines to get the diffraction resolution
                    line = loop_over_next_N_lines(text_file, 5).split()
                    ln += 5

                    if line[0] == "predict_refine/det_shift":
                        tmpframe["det_shift_x"] = line[3]
                        tmpframe["det_shift_y"] = line[6]
                        line = loop_over_next_N_lines(text_file, 1).split()
                        ln += 1

                    tmpframe["diffraction_resolution_limit [nm^-1]"] = np.float(line[2])
                    tmpframe["diffraction_resolution_limit [A]"] = np.float(line[5])

                    # get the number of predicted reflections
                    num_reflections = np.int(text_file.readline().split()[-1])
                    tmpframe["num_predicted_reflections"] = num_reflections

                    # skip a few lines
                    line = loop_over_next_N_lines(text_file, 4)
                    ln += 4
                    # get the predicted reflections
                    if num_reflections > 0:
                        reflections_pos = []
                        for tmpc in range(num_reflections):
                            # read as:
                            # h    k    l          I   sigma(I)       peak background  fs/px  ss/px
                            line = np.asarray(text_file.readline().split()[:9])
                            # append only:   fs/px  ss/px  I sigma(I)
                            reflections_pos.append(line[[7, 8, 3, 4, 0, 1, 2]])
                            ln += 1
                        tmpframe["predicted_reflections"] = np.asarray(
                            reflections_pos
                        ).astype(np.float)
                    # continue reading
                    line = text_file.readline()
                    ln += 1

                # Add the frame to the series, using the frame index as key
                series[im_num] = tmpframe

            # condition to exit the while true reading cycle
            if "" == line:
                # print("file finished")
                break

    # return the series
    return series


def dictionary_parsed_to_h5(parsed_stream, outputfile):
    """Save a nested dictionary to h5."""
    idx_frames = np.asarray([fr for fr in parsed_stream.keys()])[
        np.where(np.asarray([len(fr.keys()) for fr in parsed_stream.values()]) == 13)[0]
    ]

    indexed_frames = dict({fr: parsed_stream[fr] for fr in idx_frames})
    save_dict_to_hdf5(indexed_frames, outputfile)

def lattice_param_to_matrix(a, b, c, alpha, beta, gamma):
    """
    Convert lattice parameters to a matrix representation.

    Parameters
    ----------
    a : float
        Length of lattice vector a.
    b : float
        Length of lattice vector b.
    c : float
        Length of lattice vector c.
    alpha : float
        Angle between b and c in degrees.
    beta : float
        Angle between a and c in degrees.
    gamma : float
        Angle between a and b in degrees.

    Returns
    -------
    base_matrix : numpy.ndarray
        (3, 3) matrix representing the base vectors of the lattice.

    Examples
    --------
    >>> lattice_param_to_matrix(2, 2, 2, 90, 90, 90)
    array([[2., 0., 0.],
           [0., 2., 0.],
           [0., 0., 2.]])
    """
    
    alpha, beta, gamma = np.deg2rad((alpha, beta, gamma))
    a_ = np.array([a, 0, 0])
    b_ = np.array([b*np.cos(gamma), b*np.sin(gamma), 0])
    x = np.cos(beta)
    y = (np.cos(alpha) - np.cos(beta)*np.cos(gamma)) / np.sin(gamma)
    z = np.sqrt(np.sin(beta)**2 - y**2)
    c_ = np.array([c*x, c*y, c*z])
    return np.array([a_, b_, c_])

# Define a pythonic object to handle streams
class streamfile_parser:
    def __init__(self, streamfile):
        self.streamfile = streamfile
        self.parsed = self.parse_stream()
        (
            self.beam_center_x,
            self.beam_center_y,
            self.nx,  # detector size x
            self.ny,  # detector size y
            self.clen,  # detector distance
            self.wavelength,  # beam wavelength
            self.cellpdb,  # 3 lengths, 3 angles
        ) = self.get_experiment_info()
        # convert lattice parameters to a matrix representation
        self.cell_matrix = lattice_param_to_matrix(*self.cellpdb)
        # get the reciprocal cell matrix
        self.reciprocal_cell_matrix = np.linalg.inv(self.cell_matrix).T

    def parse_stream(self):
        return stream_to_dictionary(self.streamfile)
    
    def get_experiment_info(self):
        import re
        """
        Get same info about the experiment from the stream file
        """
        posx = posy = nx = ny = clen = photon_energy = None
        cell = [0] * 6

        def is_number(string):
            try:
                float(string)
                return True
            except ValueError:
                return False

        try:
            with open(self.streamfile, "r") as f:
                for line_num, line in enumerate(f):
                    if "corner_x" in line and is_number(line.split()[2]):
                        posx = float(line.split()[2])
                    elif "corner_y" in line and is_number(line.split()[2]):
                        posy = float(line.split()[2])
                    elif "max_fs" in line and is_number(line.split()[2]):
                        nx = int(line.split()[2]) + 1
                    elif "max_ss" in line and is_number(line.split()[2]):
                        ny = int(line.split()[2]) + 1
                    elif "clen" in line and is_number(line.split()[2]):
                        clen = float(line.split()[2]) * 1000
                    elif "photon_energy_eV" in line or "photon_energy" in line:
                        energy_values = [float(i) for i in line.split() if is_number(i)]
                        if energy_values:
                            photon_energy = 12398.42 / np.float(max(energy_values))
                    
                    cell_pattern = re.compile(r"(?P<param>[abc]|al|be|ga)\s*=\s*(?P<value>\d+\.\d+)\s*(?P<unit>[Aa]|[Dd]eg)")
                    if line.startswith("a") or line.startswith("b") or line.startswith("c") or line.startswith("al") or line.startswith("be") or line.startswith("ga"):
                        match = cell_pattern.search(line)
                        if match:
                            param = match.group("param")
                            value = float(match.group("value"))
                            if param == "a":
                                cell[0] = value
                            elif param == "b":
                                cell[1] = value
                            elif param == "c":
                                cell[2] = value
                            elif param == "al":
                                cell[3] = value
                            elif param == "be":
                                cell[4] = value
                            elif param == "ga":
                                cell[5] = value

                    if all(param is not None for param in (posx, posy, nx, ny, clen, photon_energy)) and all(val != 0 for val in cell):
                        break

        except (ValueError, IndexError):
            raise ValueError(f"Invalid value encountered in line: {line.strip()}")

        missing_params = []
        if posx is None:
            missing_params.append("posx")
            posx = -33
        if posy is None:
            missing_params.append("posy")
            posy = -33
        if nx is None:
            missing_params.append("nx")
            nx = -33
        if ny is None:
            missing_params.append("ny")
            ny = -33
        if clen is None:
            missing_params.append("clen")
            clen = -33
        if photon_energy is None:
            missing_params.append("photon_energy")
            photon_energy = -33
        
        if any(val == 0 for val in cell):
            missing_cell_params = [f"{param} ({index})" for index, param in enumerate(['a', 'b', 'c', 'al', 'be', 'ga']) if cell[index] == 0]
            print(f"Warning: Missing cell parameter(s) {', '.join(missing_cell_params)} from the stream file.")
            for index, val in enumerate(cell):
                if val == 0:
                    cell[index] = -33

        if missing_params:
            print(f"Warning: Missing parameter(s) {', '.join(missing_params)} from the stream file.")

        print(f"posx {posx}, posy {posy}, nx{nx }, ny {ny}, clen {clen}, photon_energy {photon_energy}")
        print(f"cell: {cell}")

        cell = np.array(cell)
        return abs(posx), abs(posy), nx, ny, clen, photon_energy, cell


    # def get_experiment_info(self):
    #     """
    #     get same info about the experiment from the stream file
    #     """
    #     proc = subprocess.Popen(
    #         "grep corner_x {}".format(self.streamfile),
    #         stdout=subprocess.PIPE,
    #         shell=True,
    #     )
    #     (out, err) = proc.communicate()
    #     out = out.decode("UTF-8")
    #     posx = float(out.split()[2])

    #     proc = subprocess.Popen(
    #         "grep corner_y {}".format(self.streamfile),
    #         stdout=subprocess.PIPE,
    #         shell=True,
    #     )
    #     (out, err) = proc.communicate()
    #     out = out.decode("UTF-8")
    #     posy = float(out.split()[2])

    #     proc = subprocess.Popen(
    #         "grep max_fs {}".format(self.streamfile),
    #         stdout=subprocess.PIPE,
    #         shell=True,
    #     )
    #     (out, err) = proc.communicate()
    #     out = out.decode("UTF-8")
    #     nx = int(out.split()[2]) + 1

    #     proc = subprocess.Popen(
    #         "grep max_ss {}".format(self.streamfile),
    #         stdout=subprocess.PIPE,
    #         shell=True,
    #     )
    #     (out, err) = proc.communicate()
    #     out = out.decode("UTF-8")
    #     ny = int(out.split()[2]) + 1

    #     proc = subprocess.Popen(
    #         "grep clen {}".format(self.streamfile), stdout=subprocess.PIPE, shell=True
    #     )
    #     (out, err) = proc.communicate()
    #     out = out.decode("UTF-8")
    #     clen = float(out.split()[2]) * 1000

    #     proc = subprocess.Popen(
    #         "grep 'photon_energy' {}".format(self.streamfile),
    #         stdout=subprocess.PIPE,
    #         shell=True,
    #     )
    #     (out, err) = proc.communicate()
    #     out = out.decode("UTF-8")
    #     photon_energy = 12398.42 / np.float(
    #         max([int(i) for i in set(out.split()) if i.isnumeric()])
    #     )

    #     proc = subprocess.Popen(
    #         "grep -A11 'Begin unit cell'  {}".format(self.streamfile),
    #         stdout=subprocess.PIPE,
    #         shell=True,
    #     )
    #     (out, err) = proc.communicate()
    #     out = out.decode("UTF-8")
    #     cellstr = [
    #         line
    #         for line in out.split("\n")
    #         if len(line) > 0
    #         if len(line) > 1
    #         if line.split()[0] in {"a", "b", "c", "al", "be", "ga"}
    #     ]
    #     cell = np.array([np.float(line.split()[2]) for line in cellstr])
    #     return abs(posx), abs(posy), nx, ny, clen, photon_energy, cell

    def get_indexable_frames(self):
        """
        Returns a list of indexable frames.
        """
        return np.array(
            sorted(
                [frame for frame, info in self.parsed.items() if len(info.keys()) > 7]
            )
        )

    def get_cellslist(self):
        """
        Returns a list of reciprocal cells matrices
        """
        return np.array(
            [
                np.linalg.inv(self.parsed[frame]["reciprocal_cell_matrix"] / 10.0).T
                for frame in self.get_indexable_frames()
            ]
        )

    def get_spots_2d(self, frame):
        """
        Returns a the spots in the detector panel (px units)
        relative to the spcified the input frame
        """
        if "peaks" in self.parsed[frame].keys():
            return self.parsed[frame]["peaks"][:, :2]
        else:
            return np.array([])

    def rodrigues(self, h, phi, rot_ax):

        cp = np.cos(phi)
        sp = np.sin(phi)
        omcp = 1.0 - cp

        rot_h = np.zeros(3)

        rot_h[0] = (
            (cp + rot_ax[0] ** 2 * omcp) * h[0]
            + (-rot_ax[2] * sp + rot_ax[0] * rot_ax[1] * omcp) * h[1]
            + (rot_ax[1] * sp + rot_ax[0] * rot_ax[2] * omcp) * h[2]
        )
        rot_h[1] = (
            (rot_ax[2] * sp + rot_ax[0] * rot_ax[1] * omcp) * h[0]
            + (cp + rot_ax[1] ** 2 * omcp) * h[1]
            + (-rot_ax[0] * sp + rot_ax[1] * rot_ax[2] * omcp) * h[2]
        )
        rot_h[2] = (
            (-rot_ax[1] * sp + rot_ax[0] * rot_ax[2] * omcp) * h[0]
            + (rot_ax[1] * sp + rot_ax[1] * rot_ax[2] * omcp) * h[1]
            + (cp + rot_ax[2] ** 2 * omcp) * h[2]
        )

        return rot_h

    def get_spots_3d(self, frame):
        """
        Maps the spots in 2D into 3D points in the reciprocal space.
        """

        wavelength = self.wavelength
        # size of the pixels
        qx = 0.075000
        qy = 0.075000
        orgx = self.beam_center_x
        orgy = self.beam_center_y
        det_dist = self.clen
        det_x = np.asarray([1.0, 0.0, 0.0])
        det_y = np.asarray([0.0, 1.0, 0.0])
        resolmax = 0.1
        resolmin = 999
        starting_angle = 0
        oscillation_range = 0.0
        rot_ax = np.asarray([1.000000, 0.0, 0.0])

        # get the reference frame
        incident_beam = np.asarray([0.0, 0.0, 1.0 / wavelength])
        det_z = np.zeros(3)
        # comput z in case x,y are not perpendicular to the beam
        det_z[0] = (
            det_x[1] * det_y[2] - det_x[2] * det_y[1]
        )  # calculate detector normal -
        det_z[1] = det_x[2] * det_y[0] - det_x[0] * det_y[2]  # XDS.INP does not have
        det_z[2] = det_x[0] * det_y[1] - det_x[1] * det_y[0]  # this item.
        det_z = det_z / np.sqrt(np.dot(det_z, det_z))  # normalize (usually not req'd)

        spots = []

        for line in self.get_spots_2d(frame):
            (ih, ik, il) = (0.0, 0.0, 0.0)
            if len(line) == 4:
                (x, y, phi, intensity) = line
            elif len(line) == 7:
                (x, y, phi, intensity, ih, ik, il) = line
            elif len(line) == 3:
                (x, y, intensity) = line
                phi = 0.0
            else:
                (x, y) = line
                phi = 0.0
                intensity = 0.0

            # convert detector coordinates to local coordinate system
            r = np.asarray(
                [
                    (x - orgx) * qx * det_x[0]
                    + (y - orgy) * qy * det_y[0]
                    + det_dist * det_z[0],
                    (x - orgx) * qx * det_x[1]
                    + (y - orgy) * qy * det_y[1]
                    + det_dist * det_z[1],
                    (x - orgx) * qx * det_x[2]
                    + (y - orgy) * qy * det_y[2]
                    + det_dist * det_z[2],
                ]
            )

            # normalize scattered vector to obtain S1
            r = r / (wavelength * np.sqrt(np.dot(r, r)))
            # obtain reciprocal space vector S = S1-S0
            r = r - incident_beam

            if np.sqrt(np.dot(r, r)) > 1.0 / resolmax:
                continue  # outer resolution limit
            if np.sqrt(np.dot(r, r)) < 1.0 / resolmin:
                continue  # inner resolution limit

            # rotate
            # NB: the term "-180." (found by trial&error) seems to make it match dials.rs_mapper
            phi = (starting_angle + oscillation_range * phi - 180.0) / 180.0 * np.pi
            rot_r = self.rodrigues(r, phi, rot_ax)

            # rot_r=100.*rot_r + 100./resolmax  # ! transform to match dials.rs_mapper

            spots.append(np.hstack([rot_r, [intensity], [ih, ik, il]]))

        return np.asarray(spots)[:, :3]

    def indexed2file(self, h5file, outputname):
        """
        Write to a text file the indexing solutions in a format readable by indexamajig
        """
        with open(outputname, "w") as text_file:
            for fr, tmpcell in zip(self.get_indexable_frames(), self.get_cellslist()):
                reccel = np.array(self.get_reciprocal_basis(tmpcell)) * 10
                line = "{} //{} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f} 0 0 aP\n".format(
                    h5file, fr, *np.array(reccel).ravel()
                )
                text_file.write(line)

    def write_sorted_stream(self, output_file):
        """
        This function reads a CrystFEL stream file and extracts the header and chunks of data.
        The header and chunks are then written to a new file with the chucks sorted
        in ascending order.

        Parameters:
        output_file (str): The path of the sorted output stream to write

        Returns:
        None
        """
        chunks = {}
        header = ""
        current_chunk = None
        event_num = None  # define event_num before it is used
        with open(self.streamfile, "r") as f:
            for line in f:
                if "----- Begin chunk -----" in line:
                    current_chunk = []
                    break
                header += line
            for line in f:
                if "----- Begin chunk -----" in line:
                    current_chunk = []
                elif "----- End chunk -----" in line:
                    chunks[event_num] = current_chunk
                    current_chunk = None
                elif current_chunk is not None:
                    current_chunk.append(line)
                    match = re.search(r"Event: //(\d+)", line)
                    if match:
                        event_num = int(match.group(1))

        # write out the sorted output stream
        with open(output_file, "w") as f:
            f.write(header)
            for event_num in sorted(chunks.keys()):
                chunks[event_num].insert(0, "----- Begin chunk -----\n")
                chunks[event_num].append("----- End chunk -----\n")
                f.writelines(chunks[event_num])

    def get_reciprocal_basis(self, primal_basis, with_correction=False):
        """Finds the reciprocal basis for a given primal basis in direct space.

        Args:
            primal_basis:    a numpy array with shape (3,3) representing
                             the primal basis in the direct space.
            with_correction: a boolean flag indicating whether to apply
                             a correction to the reciprocal basis.
                             Default: False
        Returns:
            A numpy array with shape (3,3) representing the reciprocal basis.
        """
        a1 = primal_basis[:, 0]
        a2 = primal_basis[:, 1]
        a3 = primal_basis[:, 2]
        b1 = np.cross(a2, a3) / (np.dot(a1, np.cross(a2, a3)))
        b2 = np.cross(a3, a1) / (np.dot(a1, np.cross(a2, a3)))
        b3 = np.cross(a1, a2) / (np.dot(a1, np.cross(a2, a3)))
        basis = np.column_stack([b1, b2, b3])
        if with_correction:
            mask = np.array([[-1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, 1.0]])
            return basis * mask
        else:
            return basis


###################################################
# Script to be called from the command line


def parse_args():
    """Parser"""
    parser = argparse.ArgumentParser(description="Console script to parse indexamajig.")

    parser.add_argument(
        "--stream",
        default="input.stream",
        help="Streaming file to parse [e.g. input.stream]",
    )

    parser.add_argument(
        "--output",
        default="parsed_stream.h5",
        help="Parsed file, stored in hdf5 format -- only indexable frames are stored! [default: parsed_stream.h5]",
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
    else:
        args = parser.parse_args()
        # args.func(**vars(args))

        return args


def main():
    """ """
    # read from the parser
    inputs = parse_args()

    # parse the input stream
    print("Parsing {}".format(inputs.stream))
    stream_parsed = streamfile_parser(inputs.stream)

    # save the input stream to a .h5 file
    dictionary_parsed_to_h5(stream_parsed.parsed, inputs.output)
    print("Stream saved in {}".format(inputs.output))


if __name__ == "__main__":
    sys.exit(main())
