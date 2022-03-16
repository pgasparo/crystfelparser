"""Console script for crystfelparser."""
import argparse
import string
import sys

#from crystfelparser import crystfelparser


###### Functions to be moved elsewhere ############

from collections import defaultdict
import numpy as np
import h5py
import os


# ----- From dictionary to h5 and viceversa ------

def save_dict_to_hdf5(dic, filename):
    """
    ....
    """
    with h5py.File(filename, 'w') as h5file:
        recursively_save_dict_contents_to_group(h5file, '/', dic)

def recursively_save_dict_contents_to_group(h5file, path, dic):
    """
    ....
    """
    for key, item in dic.items():
        if isinstance(item, (np.ndarray, np.int, np.float, str, bytes)):
            h5file[path + str(key)] = item
        elif isinstance(item, dict):
            recursively_save_dict_contents_to_group(h5file, path + str(key) + '/', item)
        else:
            print(item)
            raise ValueError('Cannot save %s type'%type(item))

def load_dict_from_hdf5(filename):
    """
    ....
    """
    with h5py.File(filename, 'r') as h5file:
        return recursively_load_dict_contents_from_group(h5file, '/')

def recursively_load_dict_contents_from_group(h5file, path):
    """
    ....
    """
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item[()]
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return ans

# ----- end dictionary to h5 and viceversa -------

# ----- main parser functions --------------------

def stream_to_dictionary(streamfile):
    """ Main parser function.
    """
    series = defaultdict(dict)
    series = dict()

    def loop_over_next_N_lines(file, n_lines):
        
        for cnt_tmp in range(n_lines):
            line=file.readline()
            
        return line


    c=0
    with open(streamfile, 'r') as text_file:
        #for ln,line in enumerate(text_file):
        ln=-1
        while True:
            ln+=1
            line=text_file.readline()
            #if any(x in ["Begin","chunk"] for x in line.split()):
            if "Begin chunk" in line:            
                # create a temporary dictionary to store the output for a frame
                # tmpframe = defaultdict(int)
                tmpframe = dict()
                # set a counter given the number of lines coming after
                cnt=0
                
                # loop over the next 3 lines to get the index of the image
                # line 2 and 3 are where it is stored the image number
                line=loop_over_next_N_lines(text_file, 3)
                ln+=3
                # save the image index and save it as zero-based
                im_num=np.int(line.split()[-1])-1
                tmpframe["Image serial number"]=im_num
                
                # loop over the next 2 lines to see if the indexer worked
                line=loop_over_next_N_lines(text_file, 2)
                ln+=2
                # save who indexed the image
                indexer_tmp=line.split()[-1]
                # if indexed, there is an additional line here
                npeaks_lines=6
                if indexer_tmp=="none" :
                    npeaks_lines=5
                tmpframe["indexed_by"]=indexer_tmp
                
                
                ##### Get the STRONG REFLEXTIONS from the spotfinder #####
                
                # loop over the next 5/6 lines to get the number of strong reflctions
                line=loop_over_next_N_lines(text_file, npeaks_lines)
                ln+=npeaks_lines
                # get the number of peaks
                num_peaks=np.int(line.split()[-1])
                tmpframe["num_peaks"]=num_peaks
                
                # get the resolution
                line=text_file.readline()
                ln+=1
                tmpframe["peak_resolution [A]"]=np.float(line.split()[-2])
                tmpframe["peak_resolution [nm^-1]"]=np.float(line.split()[2])
                
                if num_peaks>0 :
                    # skip the first 2 lines
                    for tmpc in range(2):
                        text_file.readline()
                        ln+=1
                    
                    # get the spots
                    # fs/px, ss/px, (1/d)/nm^-1, Intensity
                    # with
                    # dim1 = ss, dim2 = fs
                    tmpframe["peaks"] = np.asarray([ 
                                            text_file.readline().split()[:4]
                                            for tmpc in range(num_peaks)
                                        ]).astype(np.float)
                
                
                ##### Get the PREDICTIONS after indexing #####
                
                if tmpframe["indexed_by"]!="none":
                    # skip the first 2 header lines
                    for tmpc in range(2):
                        text_file.readline()
                        ln+=1
                    # Get the unit cell -- as cell lengths and angles
                    line=text_file.readline().split()
                    tmpframe["Cell parameters"]=np.hstack([line[2:5],line[6:9]]).astype(np.float)
                    
                    # Get the reciprocal unit cell as a 3x3 matrix
                    reciprocal_cell=[]
                    for tmpc in range(3):
                        reciprocal_cell.append(text_file.readline().split()[2:5])
                        ln+=1
                    tmpframe["reciprocal_cell_matrix"]=np.asarray(reciprocal_cell).astype(np.float)
                
                    # Save the lattice type
                    tmpframe["lattice_type"]=text_file.readline().split()[-1]
                    ln+=1
                    
                    # loop over the next 6 lines to get the diffraction resolution
                    line=loop_over_next_N_lines(text_file, 6).split()
                    ln+=6
                    tmpframe["diffraction_resolution_limit [nm^-1]"]=np.float(line[2])
                    tmpframe["diffraction_resolution_limit [A]"]=np.float(line[5])
                    
                    # get the number of predicted reflections
                    num_reflections=np.int(text_file.readline().split()[-1])
                    tmpframe["num_predicted_reflections"]=num_reflections
                    
                    # skip a few lines
                    line=loop_over_next_N_lines(text_file, 4)
                    ln+=4
                    # get the predicted reflections
                    if num_reflections>0 :
                        reflections_pos=[]
                        for tmpc in range(num_reflections):
                            # read as:
                            # h    k    l          I   sigma(I)       peak background  fs/px  ss/px
                            line=np.asarray(text_file.readline().split()[:9])
                            # append only:   fs/px  ss/px  I sigma(I)
                            reflections_pos.append(line[[7,8,3,4]])
                            ln+=1
                        tmpframe["predicted_reflections"]=np.asarray(reflections_pos).astype(np.float)
                    # continue reading
                    line=text_file.readline()
                    ln+=1                
                    
                ### Add the frame to the series, using the frame index as key
                series[im_num]=tmpframe
                
            # condition to exit the while true reading cycle
            if ("" == line):
                #print("file finished")
                break

    # return the series
    return series


def dictionary_parsed_to_h5(parsed_stream,outputfile):
    """ Save to h5. """
    idx_frames=np.asarray([fr for fr in parsed_stream.keys()])[  np.where(np.asarray([len(fr.keys()) for fr in parsed_stream.values()])==13)[0]  ]

    indexed_frames=dict({fr: parsed_stream[fr] for fr in idx_frames})
    save_dict_to_hdf5(indexed_frames, outputfile)

# ----- end main parser functions ----------------


###################################################

def parse_args():
    """Parser"""
    parser = argparse.ArgumentParser(description='Console script to parse indexamajig.')

    parser.add_argument(
        '--stream', default="input.stream", help='Streaming file to parse [e.g. input.stream]'
    )

    parser.add_argument(
        '--output',
        default="parsed_stream.h5",
        help='Parsed file, stored in hdf5 format -- only indexable frames are stored! [default: parsed_stream.h5]',
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
    else:
        args = parser.parse_args()
        #args.func(**vars(args))

        return args


def main():
    """ """
    # read from the parser
    inputs=parse_args()

    # parse the input stream
    print("Parsing {}".format(inputs.stream))
    parsed_stream=stream_to_dictionary(inputs.stream)

    # save the input stream to a .h5 file
    dictionary_parsed_to_h5(parsed_stream,inputs.output)
    print("Indexable frames saved in {}".format(inputs.output))

if __name__ == "__main__":
    sys.exit(main())

