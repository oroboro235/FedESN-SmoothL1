# This file is used to store the configuration of the project.

# datasets

tsr_dataset_names = ["mg_t17", "lorenz"]
tsc_dataset_names = ["har", "char", "ucr", "uea"]


class path:
    cache_path = "./cache/"
    pics_path = "./result/pic/"
    data_root = "./raw/"
    log_path = "./result/log/"
    model_path = "./result/model/"

    def __str__(self):
        return "cache_path: " + self.cache_path + "\n" + \
               "pics_path: " + self.pics_path + "\n" + \
               "data_root: " + self.data_root + "\n" + \
               "log_path: " + self.log_path + "\n" + \
               "model_path: " + self.model_path + "\n"
    

def get_data_path(dataset_name):
        if dataset_name in tsr_dataset_names:
            return path.data_root + "tsr/" + dataset_name + "/"
        elif dataset_name in tsc_dataset_names:
            if dataset_name == "ucr":
                 return path.data_root + "tsc/UCR_univariate/"
            if dataset_name == "uea":
                 return path.data_root + "tsc/UEA_multivariate/"
            return path.data_root + "tsc/" + dataset_name + "/"
        else:
            raise ValueError("Invalid dataset name: " + dataset_name)
        

# echo state network settings

# reservoir parameters
res_params = {

}

# federated learning settings
fed_params = {
     
}


# get a string of all configurations
def get_config_str():
    config_str = ""
    
    # path
    config_str += "path: \n" + str(path()) + "\n"

    # reservoir parameters
    config_str += "res_params: \n" + str(res_params) + "\n"

    # federated learning settings
    config_str += "fed_params: \n" + str(fed_params) + "\n"

    return config_str


if __name__ == "__main__":
    print(get_config_str())