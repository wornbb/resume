import numpy as np
from loading import *
from GLSP import *
from eagle import *
import pickle
from plot_generation import *
if __name__ == "__main__":
    train_trigger = True
    plot_trigger = True

    # data location
    f_list = [
        "C:\\Users\\Yi\Desktop\\analysis_pred\\pyscripts\\" + "blackscholes2c" + ".gridIR",
        "C:\\Users\\Yi\Desktop\\analysis_pred\\pyscripts\\" + "bodytrack2c" + ".gridIR",
        "C:\\Users\\Yi\Desktop\\analysis_pred\\pyscripts\\" + "freqmine2c"+ ".gridIR",
        "C:\\Users\\Yi\Desktop\\analysis_pred\\pyscripts\\" + "facesim2c"+ ".gridIR",
        ]
    fname = "C:\\Users\\Yi\Desktop\\analysis_pred\\pyscripts\\" + "blackscholes2c" + ".gridIR"
    flp = Path(r"C:\Users\Yi\Desktop\analysis_pred\pyscripts").joinpath("2c.png")
    if train_trigger:
        # loading data
        regression_data = regression_training_data_factory(load_flist=[fname], lines_to_read=2000)
        [x, y] = regression_data.generate()
        print("Loading Completed")
        # group lasso
        gl = gl_model(pred_str=0)
        gl.fit(x)
        pickle.dump(gl, open("gl.pred_0.model", "wb"))
        print("GL completed")
        # # eagle eye with segmentation
        # ee_segmented = ee_model(flp_fname=flp, gridIR=fname, pred_str=0, placement_mode="selected_IC")
        # ee_segmented.fit(x)
        # pickle.dump(ee_segmented, open("ee.segmented.pred_0.model", "wb"))
        # print("segmented EE completed")
        # # eagle eye vanilla
        # ee_vanilla = ee_model(flp_fname=flp, gridIR=fname, pred_str=0, segment_trigger=False)
        # ee_vanilla.fit(x)
        # pickle.dump(ee_vanilla, open("ee.vanilla.pred_0.model", "wb"))
        # print("vanilla EE completed")

    if plot_trigger:
        gl_plot = benchmark_factory(model_flist="gl.pred_0.model", data_list=[fname], exp_name="gl.pred_0", mode="regression",pred_str_list=[0], flp=flp, lines_to_read=20000 )
        gl_plot.benchmarking()
        # ee_segmented_plot = benchmark_factory(model_flist=["ee.segmented.pred_0.model"], data_list=f_list, exp_name="ee.segmented.pred_0", mode="regression",pred_str_list=[0], flp=flp )
        # ee_segmented_plot.benchmarking()
        # ee_vanilla_plot = benchmark_factory(model_flist=["ee.vanilla.pred_0.model"], data_list=f_list, exp_name="ee.vanilla.pred_0", mode="regression",pred_str_list=[0], flp=flp )
        # ee_segmented_plot.benchmarking()
