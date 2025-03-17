import logging
import os
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.widgets import SpanSelector
from scipy.signal import savgol_filter
import joblib  # if global_model.pkl exists

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)s] %(asctime)s - %(message)s'
)

class VT1SynergyApp:
    def __init__(self, master):
        self.master = master
        self.master.title("VT1 Synergy (Local & Global lines visible + HR on third axis)")

        # Top row
        top_frame = tk.Frame(master)
        top_frame.pack(pady=5, fill=tk.X)

        self.load_button = tk.Button(top_frame, text="Load File (CSV/XLSX)", command=self.load_data)
        self.load_button.pack(side=tk.LEFT, padx=5)

        self.run_button = tk.Button(top_frame, text="Run Synergy", command=self.run_synergy_analysis, state=tk.DISABLED)
        self.run_button.pack(side=tk.LEFT, padx=5)

        # Filter & Fitness Frame
        param_frame = tk.LabelFrame(master, text="Filter / Skip / Fitness")
        param_frame.pack(pady=5, fill=tk.X)

        tk.Label(param_frame, text="Ignore Start (sec):").grid(row=0, column=0, sticky='e', padx=5, pady=2)
        self.skip_start_var = tk.StringVar(value="30")
        tk.Spinbox(param_frame, textvariable=self.skip_start_var, from_=0, to=300, increment=10, width=5).grid(row=0, column=1, padx=5, pady=2)

        tk.Label(param_frame, text="Ignore End (sec):").grid(row=0, column=2, sticky='e', padx=5, pady=2)
        self.skip_end_var = tk.StringVar(value="30")
        tk.Spinbox(param_frame, textvariable=self.skip_end_var, from_=0, to=300, increment=10, width=5).grid(row=0, column=3, padx=5, pady=2)

        tk.Label(param_frame, text="Artifact Z-Threshold:").grid(row=1, column=0, sticky='e', padx=5, pady=2)
        self.artifact_var = tk.StringVar(value="4.0")
        tk.Spinbox(param_frame, textvariable=self.artifact_var, from_=0.1, to=10.0, increment=0.1, width=5).grid(row=1, column=1, padx=5, pady=2)
        tk.Label(param_frame, text="(rMSSD/HF outlier removal)").grid(row=1, column=2, columnspan=2, sticky='w')

        tk.Label(param_frame, text="Filter (rMSSD/HF):").grid(row=2, column=0, sticky='e', padx=5, pady=2)
        self.filter_type_var = tk.StringVar(value="None")
        tk.OptionMenu(param_frame, self.filter_type_var, "None", "Savitzky-Golay", "Moving Average"
                     ).grid(row=2, column=1, padx=5, pady=2)

        tk.Label(param_frame, text="Filter Window:").grid(row=3, column=0, sticky='e', padx=5, pady=2)
        self.window_var = tk.StringVar(value="11")
        tk.Spinbox(param_frame, textvariable=self.window_var, from_=3, to=101, increment=2, width=5
                   ).grid(row=3, column=1, padx=5, pady=2)
        tk.Label(param_frame, text="SG Polyorder:").grid(row=3, column=2, sticky='e', padx=5, pady=2)
        self.polyorder_var = tk.StringVar(value="2")
        tk.Spinbox(param_frame, textvariable=self.polyorder_var, from_=1, to=5, increment=1, width=5
                   ).grid(row=3, column=3, padx=5, pady=2)

        # Fitness
        tk.Label(param_frame, text="Fitness Level:").grid(row=4, column=0, sticky='e', padx=5, pady=2)
        self.fitness_var = tk.StringVar(value="Moderate")
        tk.OptionMenu(param_frame, self.fitness_var, "Low","Moderate","High"
                     ).grid(row=4, column=1, padx=5, pady=2)

        tk.Label(param_frame, text="± BPM tolerance:").grid(row=4, column=2, sticky='e', padx=5, pady=2)
        self.hr_tolerance_var = tk.StringVar(value="5")
        tk.Spinbox(param_frame, textvariable=self.hr_tolerance_var, from_=0, to=30, increment=1, width=5
                   ).grid(row=4, column=3, padx=5, pady=2)

        self.filter_button = tk.Button(param_frame, text="Apply / Show Raw vs Filtered", command=self.apply_filter)
        self.filter_button.grid(row=5, column=0, columnspan=4, padx=5, pady=5)

        # Bottom row
        bottom_frame = tk.Frame(master)
        bottom_frame.pack(pady=5, fill=tk.X)

        self.save_button = tk.Button(bottom_frame, text="Save Region to Global Data",
                                     command=self.save_region, state=tk.DISABLED)
        self.save_button.pack(side=tk.LEFT, padx=5)

        self.info_label = tk.Label(bottom_frame, text="Load a file to begin.")
        self.info_label.pack(side=tk.LEFT, padx=5)

        # Data
        self.df_original = None
        self.df_filtered = None
        self.time_col = None
        self.selected_xmin = None
        self.selected_xmax = None

        # Attempt load global model
        self.global_model = None
        self.load_global_model()

        # Fitness Ranges
        self.fitness_ranges= {
            "Low": (110,130),
            "Moderate": (125,150),
            "High": (140,160)
        }

    def load_global_model(self):
        if os.path.exists("global_model.pkl"):
            try:
                self.global_model= joblib.load("global_model.pkl")
                logging.info("Loaded global_model.pkl successfully.")
            except Exception as e:
                logging.warning(f"Failed to load global_model.pkl => {e}")
        else:
            logging.info("No global_model.pkl found => no global synergy predictions")

    # ===================== LOAD USER FILE =====================
    def load_data(self):
        file_path= filedialog.askopenfilename(
            title="Select Excel or CSV File",
            filetypes=[("Excel/CSV files","*.xlsx *.xls *.csv")]
        )
        if not file_path:
            return
        try:
            if file_path.lower().endswith((".xlsx",".xls")):
                df= pd.read_excel(file_path)
            else:
                df= pd.read_csv(file_path)

            # bulletproof synonyms
            synonyms= {
                "Time": ["time","Time","timestamp","Timestamp","datetime","Date","Timer"],
                "rMSSD":["rMSSD","rmssd","RMSSD"],
                "alpha1":["alpha1","Alpha1","ALPHA1","dfa alpha1","dfaAlpha1"],
                "HF":   ["hf","HF","highfreq","HighFreq","high_frequency"],
                "HR":   ["hr","HR","heart_rate","HeartRate","Heart Rate"]  # optional
            }
            req_main= ["Time","rMSSD","alpha1","HF"]
            missing_cols=[]

            for canon, syns in synonyms.items():
                found= self.find_col_name_in_df(syns, df.columns)
                if found:
                    df.rename(columns={found: canon}, inplace=True)
                else:
                    if canon in req_main:
                        missing_cols.append(canon)
                    else:
                        logging.info(f"Optional col {canon} not found => no HR if missing.")
            if missing_cols:
                raise ValueError(f"Missing required columns => {missing_cols}")

            df= df.dropna(subset=["Time","rMSSD","alpha1","HF"]).reset_index(drop=True)
            df= self.robust_timeparse(df)

            self.df_original= df
            self.df_filtered= None
            self.time_col= None
            self.selected_xmin= None
            self.selected_xmax= None

            self.run_button.config(state=tk.NORMAL)
            self.save_button.config(state=tk.DISABLED)
            self.info_label.config(text="File loaded. Filter optional, then run synergy.")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            logging.error(f"Failed to load data: {e}", exc_info=True)

    def find_col_name_in_df(self, synonyms, dfcols):
        norm_syn = [s.lower().replace("_","").replace(" ","") for s in synonyms]
        for c in dfcols:
            c_norm= c.lower().replace("_","").replace(" ","")
            if c_norm in norm_syn:
                return c
        return None

    def robust_timeparse(self, df):
        if "Time" not in df.columns:
            return df
        try:
            parsed= pd.to_datetime(df["Time"], errors="raise")
            if pd.api.types.is_datetime64_any_dtype(parsed):
                t0= parsed.iloc[0]
                df["TimeSec"]= (parsed - t0).dt.total_seconds()
                return df
        except:
            pass
        # fallback => parse H:M:S or row index
        sec_list=[]
        for i,val in enumerate(df["Time"]):
            val_str= str(val)
            splitted= val_str.split(":")
            if len(splitted)==3:
                try:
                    hh= float(splitted[0])
                    mm= float(splitted[1])
                    ss= float(splitted[2])
                    total_s= hh*3600+ mm*60+ ss
                    sec_list.append(total_s)
                except:
                    sec_list.append(i)
            else:
                sec_list.append(i)
        df["TimeSec"]= sec_list
        return df

    # ===================== APPLY FILTER =====================
    def apply_filter(self):
        if self.df_original is None:
            messagebox.showwarning("No data","Load a file first.")
            return

        df= self.df_original.copy()
        skip_start= float(self.skip_start_var.get())
        skip_end= float(self.skip_end_var.get())
        max_t= df["TimeSec"].iloc[-1]
        df= df[(df["TimeSec"]>= skip_start)&(df["TimeSec"]<= max_t - skip_end)]
        df= df.reset_index(drop=True)
        if len(df)<2:
            messagebox.showwarning("Warning","Skipping too much => insufficient data.")
            return

        # artifact
        z_thr= float(self.artifact_var.get())
        for c in ["rMSSD","HF"]:
            if c in df.columns:
                data= df[c].values.astype(float)
                med= np.median(data)
                mad= np.median(np.abs(data- med)) or 1e-9
                z_robust= 0.6745*(data- med)/ mad
                outliers= np.where(np.abs(z_robust)> z_thr)[0]
                data[outliers]= np.nan
                df[c]= data
                df[c]= df[c].interpolate(method="linear").fillna(method="bfill").fillna(method="ffill")

        # smoothing
        ftype= self.filter_type_var.get()
        wlen= int(self.window_var.get())
        poly= int(self.polyorder_var.get())

        if len(df)<3:
            messagebox.showwarning("Warning","Not enough data after artifact removal.")
            return
        if wlen> len(df):
            wlen= len(df) if len(df)%2==1 else len(df)-1
        if wlen<3:
            wlen=3

        if ftype=="Savitzky-Golay":
            df["rMSSD"]= savgol_filter(df["rMSSD"], wlen, poly)
            df["HF"]= savgol_filter(df["HF"], wlen, poly)
        elif ftype=="Moving Average":
            win= min(wlen, len(df))
            df["rMSSD"]= df["rMSSD"].rolling(win, center=True, min_periods=1).mean()
            df["HF"]= df["HF"].rolling(win, center=True, min_periods=1).mean()

        self.df_filtered= df
        self.show_raw_vs_filtered()

    def show_raw_vs_filtered(self):
        df_raw= self.df_original
        df_filt= self.df_filtered
        if df_raw is None or df_filt is None:
            return

        has_hr= ("HR" in df_raw.columns)
        cols=4 if has_hr else 3

        fig, axs= plt.subplots(nrows=2, ncols=cols, figsize=(12,6), sharex=True)
        fig.suptitle("Raw vs Filtered Data")

        # top => raw
        axs[0,0].plot(df_raw["TimeSec"], df_raw["rMSSD"], color="blue")
        axs[0,0].set_title("Raw rMSSD")

        axs[0,1].plot(df_raw["TimeSec"], df_raw["alpha1"], color="orange")
        axs[0,1].set_title("Raw alpha1")

        axs[0,2].plot(df_raw["TimeSec"], df_raw["HF"], color="green")
        axs[0,2].set_title("Raw HF")

        if has_hr and cols==4:
            axs[0,3].plot(df_raw["TimeSec"], df_raw["HR"], color="red")
            axs[0,3].set_title("Raw HR")

        # bottom => filtered
        axs[1,0].plot(df_filt["TimeSec"], df_filt["rMSSD"], color="blue")
        axs[1,0].set_title("Filtered rMSSD")

        axs[1,1].plot(df_filt["TimeSec"], df_filt["alpha1"], color="orange")
        axs[1,1].set_title("alpha1 (unchanged)")

        axs[1,2].plot(df_filt["TimeSec"], df_filt["HF"], color="green")
        axs[1,2].set_title("Filtered HF")

        if has_hr and cols==4:
            axs[1,3].plot(df_filt["TimeSec"], df_filt["HR"], color="red")
            axs[1,3].set_title("HR (unchanged)")

        for ax in axs[-1,:]:
            ax.set_xlabel("Time (sec)")
        plt.tight_layout()
        plt.show()

        self.run_button.config(state=tk.NORMAL)
        self.info_label.config(text="Filtering done. Now run synergy if you want.")

    # ===================== RUN SYNERGY =====================
    def run_synergy_analysis(self):
        if self.df_original is None:
            messagebox.showwarning("No data","Load a file first.")
            return

        df= self.df_filtered if self.df_filtered is not None else self.df_original
        if len(df)<5:
            messagebox.showwarning("Warning","Not enough data for synergy.")
            return

        self.time_col= df["TimeSec"].values
        synergy_score= self.partial_synergy_score(df)

        synergy_pred_global= None
        if self.global_model is not None:
            feats= self.build_features(df)
            synergy_pred_global= self.global_model.predict(feats)

        # Plot synergy with a third axis for HR, a second axis for synergy
        self.plot_synergy(df, synergy_score, synergy_pred_global)

        self.save_button.config(state=tk.DISABLED)
        self.selected_xmin= None
        self.selected_xmax= None
        self.info_label.config(text="Synergy done. Drag a region => Save Region to Global Data")

    def partial_synergy_score(self, df):
        """
        Weighted partial synergy approach with Fitness-based HR logic.
        rMSSD slope=2, HF slope=1, alpha=1, HR=2 if in range
        threshold => synergy=1 if >= 0.5
        """
        r_arr= df["rMSSD"].values
        a_arr= df["alpha1"].values
        h_arr= df["HF"].values
        hr_arr= df["HR"].values if "HR" in df.columns else None
        N= len(r_arr)

        synergy_score= np.zeros(N, dtype=float)

        w_rMSSD=2.0
        w_HR=2.0
        w_HF=1.0
        w_alpha=1.0
        total_w= w_rMSSD + w_HR + w_HF + w_alpha

        # slope
        slope_r= np.full(N, np.nan)
        slope_h= np.full(N, np.nan)
        wsize=3
        for i in range(N-wsize):
            slope_r[i+wsize//2]= (r_arr[i+wsize-1]- r_arr[i])/ wsize
            slope_h[i+wsize//2]= (h_arr[i+wsize-1]- h_arr[i])/ wsize

        # fitness
        fit_lvl= self.fitness_var.get()
        hr_low, hr_high= (125,150)
        if fit_lvl in self.fitness_ranges:
            hr_low, hr_high= self.fitness_ranges[fit_lvl]
        tol= float(self.hr_tolerance_var.get())
        hr_min= hr_low - tol
        hr_max= hr_high + tol

        for i in range(N):
            sc=0.0
            # rMSSD slope
            if not np.isnan(slope_r[i]):
                if slope_r[i]< -0.2:
                    sc+= w_rMSSD
                elif slope_r[i]<0:
                    sc+= w_rMSSD*0.5

            # HR
            if hr_arr is not None:
                hr_val= hr_arr[i]
                if not np.isnan(hr_val):
                    if hr_min<= hr_val<= hr_max:
                        sc+= w_HR
                    elif (hr_val>=(hr_min-5) and hr_val< hr_min) or (hr_val> hr_max and hr_val<= (hr_max+5)):
                        sc+= w_HR*0.5

            # HF slope
            if not np.isnan(slope_h[i]):
                if slope_h[i]< -0.05:
                    sc+= w_HF
                elif slope_h[i]<0:
                    sc+= w_HF*0.5

            # alpha
            val_a= a_arr[i]
            if val_a<0.8:
                sc+= w_alpha
            elif val_a<1.2:
                sc+= w_alpha*0.5

            synergy_score[i]= sc/ total_w

        return synergy_score

    def build_features(self, df):
        """
        shape(N,6) => [rMSSD, slope_rMSSD, HR, HF, slope_HF, alpha1]
        """
        N= len(df)
        r_arr= df["rMSSD"].values
        a_arr= df["alpha1"].values
        h_arr= df["HF"].values
        if "HR" in df.columns:
            hr_arr= np.nan_to_num(df["HR"].values, nan=0.0)
        else:
            hr_arr= np.zeros(N)

        slope_r= np.zeros(N)
        slope_h= np.zeros(N)
        w=3
        for i in range(N-w):
            slope_r[i+w//2]= (r_arr[i+w-1]- r_arr[i])/ w
            slope_h[i+w//2]= (h_arr[i+w-1]- h_arr[i])/ w

        feats= np.column_stack((r_arr, slope_r, hr_arr, h_arr, slope_h, a_arr))
        return feats

    def plot_synergy(self, df, synergy_score, synergy_pred_global):
        """
        We do 3 axes:
          - ax  => rMSSD, HF, alpha1
          - ax2 => synergy in range 0..1
          - ax3 => HR in its BPM range
        """
        t_arr= df["TimeSec"].values
        r_arr= df["rMSSD"].values
        h_arr= df["HF"].values
        a_arr= df["alpha1"].values
        hr_arr= df["HR"].values if "HR" in df.columns else None

        fig, ax= plt.subplots(figsize=(10,5))

        # 1) Main axis => rMSSD, HF, alpha
        line_r= ax.plot(t_arr, r_arr, label="rMSSD", color="blue")
        line_h= ax.plot(t_arr, h_arr, label="HF", color="green")
        line_a= ax.plot(t_arr, a_arr, label="alpha1", color="orange")

        ax.set_xlabel("Time (sec)")
        ax.set_ylabel("rMSSD / HF / alpha1")
        ax.set_title("VT1 Synergy: Drag to label synergy region")

        # 2) second axis => synergy in [0..1]
        ax2= ax.twinx()
        ax2.set_ylim(0,1.2)  # fix synergy scale
        synergy_line= ax2.plot(t_arr, synergy_score, linestyle="--", color="purple", label="Local synergy (0-1)")

        # if we have synergy_pred_global => 0 or 1 => do a step plot
        if synergy_pred_global is not None:
            # shift them => map 0->0.1, 1->0.9
            synergy_pred_plot= synergy_pred_global*0.8+ 0.1
            global_line= ax2.step(t_arr, synergy_pred_plot, where='post', color="red", label="Global synergy pred (0 or 1)")

        # 3) third axis => HR in BPM
        ax3= ax.twinx()
        # shift the third axis to the right
        ax3.spines["right"].set_position(("axes",1.15))
        if hr_arr is not None:
            hr_min= np.nanmin(hr_arr)
            hr_max= np.nanmax(hr_arr)
            if np.isnan(hr_min) or np.isnan(hr_max):
                hr_min= 0
                hr_max= 200
            else:
                hr_min= max(0, hr_min-10)
                hr_max+= 10
            ax3.set_ylim(hr_min, hr_max)
            hr_line= ax3.plot(t_arr, hr_arr, color="gray", label="HR (bpm)")
            ax3.set_ylabel("HR (bpm)")

        # build legend
        lines_main, labels_main= ax.get_legend_handles_labels()
        lines_sy, labels_sy= ax2.get_legend_handles_labels()
        if hr_arr is not None:
            lines_hr, labels_hr= ax3.get_legend_handles_labels()
            ax3.legend(lines_main+lines_sy+lines_hr, labels_main+labels_sy+labels_hr, loc="upper right")
        else:
            ax2.legend(lines_main+lines_sy, labels_main+labels_sy, loc="upper right")

        def on_select(xmin, xmax):
            self.selected_xmin= xmin
            self.selected_xmax= xmax
            logging.info(f"User synergy region => {xmin:.1f}-{xmax:.1f}")
            self.save_button.config(state=tk.NORMAL)

        self.span= SpanSelector(ax, on_select, direction='horizontal', useblit=True,
                                props=dict(alpha=0.3, facecolor='yellow'), interactive=True)

        plt.tight_layout()
        plt.show()

    # ----------------------------------------------------------------
    # SAVE synergy-labeled region => global_labeled_data.csv
    # ----------------------------------------------------------------
    def save_region(self):
        if self.df_original is None:
            messagebox.showwarning("No data","Load a file first.")
            return
        if self.selected_xmin is None or self.selected_xmax is None:
            messagebox.showwarning("No region","Drag synergy region first.")
            return

        df= self.df_filtered if self.df_filtered is not None else self.df_original
        synergy_label= np.zeros(len(df), dtype=int)
        for i,t in enumerate(df["TimeSec"].values):
            if self.selected_xmin<= t<= self.selected_xmax:
                synergy_label[i]=1

        feats= self.build_features(df)
        if feats.shape[0]!= synergy_label.shape[0]:
            messagebox.showwarning("Mismatch","Feature array != synergy_label length.")
            return

        col_names= ["rMSSD","slope_rMSSD","HR","HF","slope_HF","alpha1","SynergyLabel"]
        out_arr= np.column_stack((feats, synergy_label))
        out_df= pd.DataFrame(out_arr, columns=col_names)

        csv_file= "global_labeled_data.csv"
        if not os.path.exists(csv_file):
            out_df.to_csv(csv_file, index=False)
            logging.info(f"Created {csv_file} with synergy-labeled samples.")
        else:
            out_df.to_csv(csv_file, mode='a', header=False, index=False)
            logging.info(f"Appended synergy-labeled data to {csv_file}.")

        self.info_label.config(text="Region saved. Re-run train_global_model.py to update global model.")
        self.save_button.config(state=tk.DISABLED)
        self.selected_xmin= None
        self.selected_xmax= None


# Main
if __name__=="__main__":
    root= tk.Tk()
    app= VT1SynergyApp(root)
    root.mainloop()
