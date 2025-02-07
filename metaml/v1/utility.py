


def write_line_to_log(log_cfg, text):
    if log_cfg['enable']:
        with open(log_cfg['log_file'], 'a') as file:
            file.write(text + "\n")
            print("Write to Log: "+text)

def write_dic_to_log(log_cfg, text_dict):
    if log_cfg['enable']:
        with open(log_cfg['log_file'], 'a') as file:
            file.write("\n")
            for key in text_dict:
                file.write(key+": "+str(text_dict[key]) + "\n")
            file.write("\n")


def get_fpga_part(fpga_name):
    if(fpga_name=="U250"):
        fpga_part = "xcu250-figd2104-2L-e"
    elif (fpga_name=="KU115"):
        fpga_part = "xcku115-flvb2104-2-i"
    elif (fpga_name=="VU9P"):
        fpga_part = "xcvu9p-flgb2104-2l-e"
    elif (fpga_name=="Z7045"):
        fpga_part = "xc7z045ffg900-2"
    elif (fpga_name=="A200"):
        fpga_part = "xc7a200tsbv484-1"
    elif (fpga_name=='Z7020'):
        fpga_part = "xc7z020clg400-1"
    else:
        print("please select a supported FPGA")
        exit()
    return fpga_part
