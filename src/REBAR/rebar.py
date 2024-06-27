import numpy as np
import pandas as pd
import scipy
import json
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


class REBAR:

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _warning(self, msg, prefix="WARNING"):
        print(prefix, '-', msg)
        # exit()

    def _error(self, msg, prefix="ERROR"):
        print(prefix, '-', msg)
        exit()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _load_dataframe(self, d, specifier=None):
        if(isinstance(d, str)):
            try:
                return pd.read_csv(d)
            except FileNotFoundError:
                self._error(f"Could not find {specifier+' ' if specifier is not None else ''}file: {d}")
            except pd.errors.ParserError:
                self._error(f"Could not parse {specifier+' ' if specifier is not None else ''}file (.csv expected): {d}")
        elif(isinstance(d, pd.DataFrame)):
            return d

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _load_config(self, d, specifier=None):
        if(isinstance(d, str)):
            try:
                with open(d) as cfg_json:
                    return json.load(cfg_json)
            except FileNotFoundError:
                self._error(f"Could not find {specifier+' ' if specifier is not None else ''}file: {d}")
            except json.decoder.JSONDecodeError:
                self._error(f"Could not parse {specifier+' ' if specifier is not None else ''}file (.json expected): {d}")
        elif(isinstance(d, pd.DataFrame)):
            return d

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def __init__(self, counts=None, samples=None, variants=None, config=None, outdir=None,
                       init_bias_prev=None, init_bias_susc=None, init_normalization_factors=None,
                       output_initial=True, output_intermediates=True, output_final=True, seed=None):

        if(seed is not None):
            np.random.seed(seed)
            self.seed = seed

        self.log    = {}
        self.outdir = outdir
        self.output_initial = output_initial
        self.output_intermediates = output_intermediates
        self.output_final = output_final

        #------------------------------
        # Load counts data and metadata for samples and variants:
        self.cfg        = self._load_config(config)
        self.samplesDf  = self._load_dataframe(samples)
        self.variantsDf = self._load_dataframe(variants)
        self.countsDf   = self._load_dataframe(counts)
        #------------------------------

        self.samplesDf           = self.samplesDf.sort_values(['assay', 'time'], ascending=[True, True], ignore_index=True)
        # - - - -
        self.assay_labels        = self.samplesDf['assay'].unique()
        self.num_assays          = len(self.assay_labels)
        self.assay_indices       = np.array(range(self.num_assays))
        # - - - -
        self.sample_labels       = self.samplesDf['sample'].values
        self.sample_times        = self.samplesDf['time'].values
        self.num_samples         = len(self.samplesDf)
        self.sample_indices      = np.array(range(self.num_samples))
        self.assay_sampleIndices = {a: self.samplesDf.loc[self.samplesDf['assay'] == self.assay_labels[a]].index.tolist() for a in self.assay_indices}

        self.num_variants        = len(self.variantsDf)
        self.variant_indices     = np.array(range(self.num_variants))
        self.control_set         = self.variantsDf['control_set'].values

        self.countsDf            = self.countsDf[self.sample_labels]  # reorder columns to match sample arrays
        self.counts_raw          = self.countsDf.to_numpy()
        self.counts_raw          = np.clip(self.counts_raw, self.cfg['pseudocount'] if 'pseudocount' in self.cfg else 0.5, np.inf)  # replace counts less than pseudocount with the pseudocount value
        self.counts_adj          = np.zeros_like(self.counts_raw)*np.nan

        self.weights = np.clip(self.counts_raw, 0, self.cfg['weight_saturation_count'] if 'weight_saturation_count' in self.cfg else np.inf)

        self.trustworthy_counts  = np.zeros(shape=(self.num_variants, self.num_assays), dtype=bool)
        min_trustworthy_count    = self.cfg['min_trustworthy_count'] if 'min_trustworthy_count' in self.cfg else 0
        max_trustworthy_count    = self.cfg['max_trustworthy_count'] if 'max_trustworthy_count' in self.cfg else np.inf
        for a in self.assay_indices:
            samples_a = self.assay_sampleIndices[a]
            mean_counts_a = np.nanmean(self.counts_raw[:, samples_a], axis=1)
            # - - - -
            if('trustworthy_drop_top' in self.cfg):
                drop_top = self.cfg['trustworthy_drop_top'] + 1
                max_trustworthy_count_a = min(max_trustworthy_count, np.partition(mean_counts_a, -drop_top)[-drop_top])
            else:
                max_trustworthy_count_a = max_trustworthy_count
            # - - - -
            self.trustworthy_counts[:, a] = (mean_counts_a >= min_trustworthy_count) & (mean_counts_a <= max_trustworthy_count_a)
        # - - - -
        self.trustworthy_susceptibility = self.trustworthy_counts.sum(axis=1) >= (self.cfg['min_num_trustworthy_assays'] if 'min_num_trustworthy_assays' in self.cfg else 0)

        #------------------------------
        # Initialize depth normalization_factors for each sample:
        if(init_normalization_factors is not None):
            self.normalization_factors = init_normalization_factors
        else:
            self.normalization_factors = np.zeros(self.num_samples) * np.nan
            for a in self.assay_indices:
                samples_a = self.assay_sampleIndices[a]
                for s in samples_a:
                    self.normalization_factors[s] = np.nanmean(self.counts_raw[self.trustworthy_counts[:,a], s])

        #------------------------------
        # Initialize bias terms to be optimized:
        if(init_bias_susc is not None):
            self.bias_susceptibilities = init_bias_susc
        else:
            bias_susc_mean  = self.cfg['init_bias_susceptibility_mean'] if 'init_bias_susceptibility_mean' in self.cfg else 0
            bias_susc_stdev = self.cfg['init_bias_susceptibility_stdev'] if 'init_bias_susceptibility_stdev' in self.cfg else 0
            self.bias_susceptibilities = np.random.normal(bias_susc_mean, bias_susc_stdev, size=self.num_variants) if bias_susc_stdev > 0 else np.full(fill_value=bias_susc_mean, shape=self.num_variants)
        # - - - - - - - -
        if(init_bias_prev is not None):
            self.bias_prevalences = init_bias_prev
        else:
            bias_prev_mean  = self.cfg['init_bias_prevalence_mean'] if 'init_bias_prevalence_mean' in self.cfg else 0
            bias_prev_stdev = self.cfg['init_bias_prevalence_stdev'] if 'init_bias_prevalence_stdev' in self.cfg else 0
            self.bias_prevalences = np.random.normal(bias_prev_mean, bias_prev_stdev, size=self.num_samples) if bias_prev_stdev > 0 else np.full(fill_value=bias_prev_mean, shape=self.num_samples)

        #------------------------------
        # Instantiate dataframes for fitnesses, intercepts, and residuals returned from log-linear fits
        self.fitnesses  = np.zeros((self.num_variants, self.num_assays))*np.nan
        self.intercepts = np.zeros((self.num_variants, self.num_assays))*np.nan
        self.residuals  = np.zeros_like(self.counts_raw)*np.nan

        #------------------------------
        # Perform initial linear fits on initial normalized counts to obtain pre-correction fitness estimates and residuals
        self.fitnesses, self.intercepts, self.residuals = self._perform_fits(unweighted=True)
        self.counts_norm = self.get_normalized_counts()
        self.counts_adj = self.get_adjusted_counts()
        self.save(label='initial', save_to_file=self.output_initial, save_raw_counts=True, save_weights=True)


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    def run(self, num_iters=5, optimize_bias_susc=True, optimize_bias_prev=True, optimize_normalization_factors=True,
             weighted_regression=True, bias_prev_bounds=None, bias_susc_bounds=None,
             method='BFGS', tol=1e-8, max_opt_iter=None):

        print(f"[ Stage 1: Inferring bias susceptibilities and bias prevalence deviations. ]")
        self.infer_bias_components(num_iters=num_iters, optimize_bias_susc=optimize_bias_susc, optimize_bias_prev=optimize_bias_prev, optimize_normalization_factors=optimize_normalization_factors,
                                    weighted_regression=weighted_regression, bias_prev_bounds=bias_prev_bounds, bias_susc_bounds=bias_susc_bounds,
                                    method=method, tol=tol, max_opt_iter=max_opt_iter)

        print(f"[ Stage 2: Inferring bias prevalence trends. ]                                    ")
        self.infer_bias_trends()

        print(f"[ Computing bias-corrected fitness estimates. ]")
        self.fitnesses, self.intercepts, self.residuals = self._perform_fits(unweighted=True)
        self.counts_norm = self.get_normalized_counts()
        self.counts_adj = self.get_adjusted_counts()
        self.save(label='final', save_to_file=self.output_final)

        print(f"[ Done. ]")


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def infer_bias_components(self, num_iters=5, optimize_bias_susc=True, optimize_bias_prev=True, optimize_normalization_factors=True,
                                weighted_regression=True, bias_prev_bounds=None, bias_susc_bounds=None,
                                method='BFGS', tol=1e-8, max_opt_iter=None):

        for i in range(1, num_iters+1):
            print(f"    Iteration #{i}                                                  ")
            self.infer_sample_bias_prevalence(optimize_bias_prev=optimize_bias_prev, optimize_normalization_factors=optimize_normalization_factors, weighted_regression=weighted_regression, bounds=bias_prev_bounds, method=method, tol=tol, maxiter=max_opt_iter)
            #----------------------
            if(optimize_bias_susc):
                self.infer_variant_bias_susceptibility(weighted_regression=weighted_regression, method=method, bounds=bias_susc_bounds, tol=tol, maxiter=max_opt_iter)
            #----------------------
            FORCE_STDEV_BIASSUSC = self.cfg['target_bias_susceptibility_stdev']
            biasScaleFactor = FORCE_STDEV_BIASSUSC / np.std(self.bias_susceptibilities)
            self.bias_susceptibilities *= biasScaleFactor
            self.bias_prevalences /= biasScaleFactor
            #----------------------
            self.save(label=f"stage1-iter{i}", save_to_file=self.output_intermediates)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def infer_sample_bias_prevalence(self, optimize_bias_prev=True, optimize_normalization_factors=True, weighted_regression=True, bounds=None, method='BFGS', tol=1e-8, maxiter=None):
        print(f"        Inferring {'bias prevalences' if optimize_bias_prev else ''}{' and ' if optimize_bias_prev and optimize_normalization_factors else ''}{'normalization factors' if optimize_normalization_factors else ''}.")
        if(not optimize_bias_prev and not optimize_normalization_factors):
            self._warning("Nothing to optimize, returning. (optimize_bias_prev==False and optimize_normalization_factors==False)")
            return

        for a, samples_a in self.assay_sampleIndices.items():

            trustworthyVariants_a   = np.where(self.trustworthy_counts[:, a] == True)[0]
            biassusc_a              = self.bias_susceptibilities[trustworthyVariants_a]
            biasprev_a              = self.bias_prevalences[samples_a]
            times_a                 = self.sample_times[samples_a]
            counts_a                = self.counts_raw[trustworthyVariants_a, :][:, samples_a]
            counts_a[counts_a == 0] = self.cfg['pseudocount_value']
            weights_a               = self.weights[trustworthyVariants_a, :][:, samples_a] if weighted_regression else np.ones_like(counts_a)
            normalization_factors_a = np.mean(counts_a, axis=0)

            #::::::::::::::::::::::::::::::::::::::::::::::::::
            def norm_residuals_over_variants(params_to_opt):
                biasprev_a_cur = params_to_opt[:len(biasprev_a_opt)] if optimize_bias_prev else biasprev_a
                normalization_factors_a_cur_freevals = params_to_opt[-len(normalization_factors_a_opt_freevals):]
                normalization_factors_a_cur          = np.concatenate([np.array([normalization_factors_a[0]]), normalization_factors_a_cur_freevals, np.array([normalization_factors_a[-1]])]) if optimize_normalization_factors else normalization_factors_a
                #-------------------------
                slopes_a, intercepts_a, residuals_a = REBAR.perform_fits(counts=counts_a, times=times_a, weights=weights_a, normalization_factors=normalization_factors_a_cur, bias_prev=biasprev_a_cur, bias_susc=biassusc_a)
                residuals_a[~np.isfinite(residuals_a)] = 0.0 # convert any nan or inf residual values to 0 before calculating norm
                values_to_norm = np.concatenate([ (weights_a * residuals_a).ravel(order='F'), (self.cfg['sample_bias_penalty']*biasprev_a_cur), [((np.std(biasprev_a_cur)-self.cfg['target_bias_prevalence_stdev'])*self.cfg['sample_bias_penalty'])] ])
                return np.linalg.norm(values_to_norm)
            #::::::::::::::::::::::::::::::::::::::::::::::::::

            normalization_factors_a_opt          = np.copy(normalization_factors_a)
            normalization_factors_a_opt_freevals = np.copy(normalization_factors_a_opt[1:-1])
            biasprev_a_opt                       = np.copy(biasprev_a)
            opt_results = scipy.optimize.minimize(norm_residuals_over_variants, x0=np.concatenate([biasprev_a_opt, normalization_factors_a_opt_freevals]), bounds=bounds, method=method, tol=tol, options=({'maxiter': maxiter} if maxiter is not None else {}))

            if(optimize_bias_prev):
                self.bias_prevalences[samples_a] = opt_results['x'][:len(biasprev_a_opt)]
            if(optimize_normalization_factors):
                self.normalization_factors[samples_a] = np.concatenate([np.array([normalization_factors_a[0]]), opt_results['x'][-len(normalization_factors_a_opt_freevals):], np.array([normalization_factors_a[-1]])])

        return

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def infer_variant_bias_susceptibility(self, variants=None, weighted_regression=True, method='BFGS', bounds=None, tol=1e-8, maxiter=None, verbose=False):
        print(f"        Inferring bias susceptibilities.")
        for i in self.variant_indices:
            if(not self.trustworthy_susceptibility[i]):
                continue

            numSamples_byAssay    = []
            times_i_byAssay       = []
            counts_i_byAssay      = []
            weights_i_byAssay     = []
            normfactors_i_byAssay = []
            biasprev_i_byAssay    = []
            for a, samples_a in self.assay_sampleIndices.items():
                if(self.trustworthy_counts[i, a] != True):
                    continue  # Do not include time series from assays where this variant is not trustworthy.
                # - - - - - - - -
                counts_a_i                  = self.counts_raw[i, samples_a]
                counts_a_i[counts_a_i == 0] = self.cfg['pseudocount_value']
                weights_a_i                 = self.weights[i, samples_a] if weighted_regression else np.ones_like(counts_a_i)
                # - - - - - - - -
                counts_i_byAssay.append(counts_a_i)
                weights_i_byAssay.append(weights_a_i)
                normfactors_i_byAssay.append(self.normalization_factors[samples_a])
                biasprev_i_byAssay.append(self.bias_prevalences[samples_a])
                times_i_byAssay.append(self.sample_times[samples_a])
                numSamples_byAssay.append(len(samples_a))

            # - - - - - - - - - -
            allVariantAssaysSameNumSamples = (len(set(numSamples_byAssay)) == 1)
            if(allVariantAssaysSameNumSamples):
                counts_i_byAssay      = np.array(counts_i_byAssay)
                weights_i_byAssay     = np.array(weights_i_byAssay)
                normfactors_i_byAssay = np.array(normfactors_i_byAssay)
                biasprev_i_byAssay    = np.array(biasprev_i_byAssay)
                times_i_byAssay       = np.array(times_i_byAssay)

            #::::::::::::::::::::::::::::::::::::::::::::::::::
            def norm_residuals_over_assays(params_to_opt):
                biassusc_i_cur = params_to_opt[0]
                #-------------------------
                values_to_norm = [[self.cfg['variant_bias_penalty']*biassusc_i_cur]]
                for a in range(len(counts_i_byAssay)):
                    slope_a_i, intercept_a_i, residuals_a_i = REBAR.perform_fits(counts=counts_i_byAssay[a], times=times_i_byAssay[a], weights=weights_i_byAssay[a], normalization_factors=normfactors_i_byAssay[a], bias_prev=biasprev_i_byAssay[a], bias_susc=biassusc_i_cur)
                    residuals_a_i[~np.isfinite(residuals_a_i)] = 0.0 # convert any nan or inf residual values to 0 before calculating norm
                    values_to_norm.append((weights_i_byAssay[a] * residuals_a_i).ravel())
                values_to_norm = np.concatenate(values_to_norm)
                return np.linalg.norm(values_to_norm)
            #::::::::::::::::::::::::::::::::::::::::::::::::::

            biassusc_i_opt = np.copy(self.bias_susceptibilities[i])
            opt_results = scipy.optimize.minimize(norm_residuals_over_assays, x0=biassusc_i_opt, bounds=bounds, method=method, tol=tol, options=({'maxiter': maxiter} if maxiter is not None else {}))

            self.bias_susceptibilities[i] = opt_results['x'][0]

        return

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def infer_bias_trends(self):
        for a, samples_a in self.assay_sampleIndices.items():
            trustworthyVariants_a        = np.where(self.trustworthy_counts[:, a] == True)[0]
            trustworthyControlVariants_a = list(set(np.where(self.control_set == True)[0]).intersection(trustworthyVariants_a))
            biasprev_a = self.bias_prevalences[samples_a]
            times_a    = self.sample_times[samples_a]
            biassusc_a = self.bias_susceptibilities[trustworthyControlVariants_a]
            # - - - - - - - -
            fitnesses_controlSet_a  = self.fitnesses[trustworthyControlVariants_a, :][:, a]
            # - - - - - - - -
            delta_f_a = fitnesses_controlSet_a - fitnesses_controlSet_a.mean() # fitness misestimates among control set
            lambda_a, v_intcpt_a = REBAR.linear_regression(x=biassusc_a, y=delta_f_a)
            # - - - - - - - -
            biasprev_a_abs = biasprev_a - lambda_a*times_a
            biasprev_a_abs = biasprev_a_abs - np.mean(biasprev_a_abs)
            self.bias_prevalences[samples_a] = biasprev_a_abs
        return

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _perform_fits(self, unweighted=False):
        # PERFORM FITS FOR ALL VARIANTS FOR ALL ASSAYS IN TURN:
        slopes     = np.zeros_like(self.fitnesses)*np.nan
        intercepts = np.zeros_like(self.intercepts)*np.nan
        residuals  = np.zeros_like(self.residuals)*np.nan
        # - - - - - - - -
        for a, samples_a in self.assay_sampleIndices.items():
            slopes[:, a], intercepts[:, a], residuals[:, samples_a] = \
                REBAR.perform_fits(counts=self.counts_raw[:, samples_a], times=self.sample_times[samples_a],
                                   weights=self.weights[:, samples_a], normalization_factors=self.normalization_factors[samples_a],
                                   bias_prev=self.bias_prevalences[samples_a], bias_susc=self.bias_susceptibilities[:],
                                   unweighted=unweighted)
        # - - - - - - - -
        return slopes, intercepts, residuals

    @staticmethod
    def perform_fits(counts, times, weights, normalization_factors, bias_prev, bias_susc, unweighted=False):
        # PERFORM FITS FOR COUNT TIMESERIES PROVIDED:
        counts       = np.array([counts]) if counts.ndim < 2 else counts
        weights      = np.array([weights]) if weights.ndim < 2 else weights
        norm_factors = np.array([normalization_factors]) if not isinstance(normalization_factors, (list, np.ndarray)) else normalization_factors
        bias_susc    = np.array([bias_susc]) if not isinstance(bias_susc, (list, np.ndarray)) else bias_susc
        bias_prev    = np.array([bias_prev]) if not isinstance(bias_prev, (list, np.ndarray)) else bias_prev
        slopes       = np.zeros(len(counts))*np.nan
        intercepts   = np.zeros(len(counts))*np.nan
        residuals    = np.zeros((len(counts), counts.shape[1]))*np.nan
        # - - - - - - - -
        for ts, timeseries in enumerate(counts):
            # - - - -
            # TODO this block should be removed
            weights_ts = np.ones_like(weights[ts]) if unweighted else weights[ts]
            # - - - - - -
            adjLogCounts_ts = np.log( counts[ts]/norm_factors * np.exp(np.outer(bias_susc[ts], bias_prev)))
            # - - - - - -
            slopes[ts], intercepts[ts] = REBAR.linear_regression(x=times, y=adjLogCounts_ts, w=weights_ts)
            predLogCounts_ts           = intercepts[ts] + slopes[ts] * times
            residuals[ts, :]           = adjLogCounts_ts - predLogCounts_ts
        # - - - - - - - -
        return (slopes, intercepts, residuals)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @staticmethod
    def linear_regression(x, y, w=None):
        xw = w*x if w is not None else x
        yw = w*y if w is not None else y
        if(not np.any(yw)):
            slope     = 0
            intercept = 0
        else:
            n = len(x)
            xw_mean   = np.sum(xw)/n
            yw_mean   = np.sum(yw)/n
            w_mean    = np.sum(w)/n if w is not None else 1
            slope     = np.sum(xw * (y - yw_mean/w_mean)) / np.sum(xw * (x - xw_mean/w_mean))
            intercept = (1/w_mean)*(yw_mean - slope*xw_mean)
        return slope, intercept

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_normalized_counts(self):
        counts_norm = self.counts_raw/self.normalization_factors
        return counts_norm

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_adjusted_counts(self):
        counts_adj = (self.counts_raw/self.normalization_factors) * (np.exp(np.outer(self.bias_susceptibilities, self.bias_prevalences)))
        return counts_adj

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def save(self, label, save_to_file=True, save_raw_counts=False, save_weights=False):
        # Update information stored in data frames:
        self.samplesDf['bias_prevalence']             = self.bias_prevalences
        self.samplesDf['normalization_factor']        = self.normalization_factors
        self.variantsDf['bias_susceptibility']        = self.bias_susceptibilities
        self.variantsDf['trustworthy_susceptibility'] = self.trustworthy_susceptibility
        for a, assay in enumerate(self.assay_labels):
            self.variantsDf[f'trustworthy_{assay}']   = self.trustworthy_counts[:, a]
        # - - - - - - -
        self.log[label] = { 'samples':         self.samplesDf.copy(),
                            'variants':        self.variantsDf.copy(),
                            'counts_normalized': pd.DataFrame(data=self.counts_norm, columns=self.sample_labels),
                            'counts_adjusted': pd.DataFrame(data=self.counts_adj, columns=self.sample_labels),
                            'fitnesses':       pd.DataFrame(data=self.fitnesses, columns=self.assay_labels),
                            'intercepts':      pd.DataFrame(data=self.intercepts, columns=self.assay_labels),
                            'residuals':       pd.DataFrame(data=self.residuals, columns=self.sample_labels) }
        if(save_raw_counts):
            self.log[label].update({'counts_raw': pd.DataFrame(data=self.counts_raw, columns=self.sample_labels)})
        if(save_weights):
            self.log[label].update({'weights': pd.DataFrame(data=self.weights, columns=self.sample_labels)})

        if(save_to_file):
            if(self.outdir is None):
                self._warning("No output directory was provided, cannot save data to file.")
            else:
                self.log[label]['samples'].to_csv(self.outdir+f"/samples_{label}.csv", index=False)
                self.log[label]['variants'].to_csv(self.outdir+f"/variants_{label}.csv", index=False)
                self.log[label]['counts_normalized'].to_csv(self.outdir+f"/counts-normalized_{label}.csv", index=False)
                self.log[label]['counts_adjusted'].to_csv(self.outdir+f"/counts-adjusted_{label}.csv", index=False)
                self.log[label]['fitnesses'].to_csv(self.outdir+f"/fitnesses_{label}.csv", index=False)
                self.log[label]['intercepts'].to_csv(self.outdir+f"/intercepts_{label}.csv", index=False)
                self.log[label]['residuals'].to_csv(self.outdir+f"/residuals_{label}.csv", index=False)
                if(save_raw_counts):
                    self.log[label]['counts_raw'].to_csv(self.outdir+f"/counts-raw_{label}.csv", index=False)
                if(save_weights):
                    self.log[label]['weights'].to_csv(self.outdir+f"/weights_{label}.csv", index=False)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @staticmethod
    def generate_synthetic_data(num_assays=1, times=np.arange(0, 5, 1), num_variants=100, num_variants_control=10,
                                fitness_mean=0, fitness_stdev=0.3, fitness_stdev_control=0,
                                bias_susceptibilties=None, bias_susc_mean=0, bias_susc_stdev=0.1,
                                bias_prevalences=None,
                                bias_trends=None, bias_trend_mean=0, bias_trend_stdev=0.3,
                                bias_intercepts=None, bias_intercept_mean=0, bias_intercept_stdev=0.3,
                                bias_deviations=None, bias_deviation_mean=0, bias_deviation_stdev=1,
                                target_density_per_variant=5000, num_pcr_cycles=10, total_reads_per_sample=1e6,
                                outdir=None, return_data=False, seed=None):

        np.random.seed(seed)

        num_samples            = num_assays * len(times)
        assays_sampleIndices   = {a: list(range(num_samples))[a*len(times) : a*len(times)+len(times)] for a in range(num_assays)}
        assay_labels           = [f'Assay{a+1}' for a in range(num_assays)]
        sample_labels          = [f'Assay{(s//len(times))+1}-T{times[s%len(times)]}' for s in range(num_samples)]

        fitnesses = np.random.normal(loc=fitness_mean, scale=fitness_stdev, size=(num_variants, num_assays))
        fitnesses[:num_variants_control, :] = np.random.normal(loc=fitness_mean, scale=fitness_stdev_control, size=(num_variants_control, num_assays))

        bias_susc = bias_susceptibilties if bias_susceptibilties is not None else np.random.normal(loc=bias_susc_mean, scale=bias_susc_stdev, size=num_variants)
        # - - - - - -
        if(bias_prevalences is not None):
            bias_prev = bias_prevalences
            bias_prev_trends     = None
            bias_prev_intercepts = None
            bias_prev_deviations = None
        else:
            bias_prev_trends     = bias_trends if bias_trends is not None else np.random.normal(loc=bias_trend_mean, scale=bias_trend_stdev, size=num_assays)
            bias_prev_intercepts = bias_intercepts if bias_intercepts is not None else np.random.normal(loc=bias_intercept_mean, scale=bias_intercept_stdev, size=num_assays)
            bias_prev_deviations = bias_deviations if bias_deviations is not None else np.random.normal(loc=bias_deviation_mean, scale=bias_deviation_stdev, size=num_samples)
            bias_prev            = np.zeros(num_samples)*np.nan
            for a, samples_a in assays_sampleIndices.items():
                bias_prev[samples_a] = (bias_prev_trends[a]*times + bias_prev_intercepts[a]) + bias_prev_deviations[samples_a]
            # biastrendsDf = bias_prev_trends.reshape((1, num_assays))
            # if(outdir is not None):
            #     pd.DataFrame(data=biastrendsDf, columns=assay_labels).to_csv(outdir+'biastrends.csv')
        # - - - - - -
        bias_effects = np.outer(bias_susc, bias_prev)

        # print("bias_prev_trends", bias_prev_trends)
        # print("bias_prev_deviations", bias_prev_deviations)
        # print("bias_prev", bias_prev)
        # print("lambda1", np.polyfit(times, bias_prev[0*len(times):1*len(times)], 1)[0],
        #       "lambda2", np.polyfit(times, bias_prev[1*len(times):2*len(times)], 1)[0],
        #       "lambda3", np.polyfit(times, bias_prev[2*len(times):3*len(times)], 1)[0])

        cellAbundances = np.zeros((num_variants, num_samples))*np.nan
        for a, samples_a in assays_sampleIndices.items():
            fitnesses_a = fitnesses[:, a]
            for tidx, s in enumerate(samples_a):
                if(tidx == 0):
                    cellAbundances[:, s] = np.random.poisson(target_density_per_variant)
                if(tidx != 0):
                    dt = times[tidx] - times[tidx-1]
                    cellAbundances_prev  = cellAbundances[:, s-1]
                    cellAbundances_xfr   = np.random.poisson(cellAbundances_prev * ((target_density_per_variant*num_variants)/np.sum(cellAbundances_prev))) if tidx > 1 else cellAbundances_prev
                    cellAbundances_cur   = np.random.poisson(cellAbundances_xfr * np.exp(fitnesses_a * dt))
                    cellAbundances[:, s] = cellAbundances_cur

        barcodeCopies = cellAbundances * 2**num_pcr_cycles * np.exp(bias_effects)
        readCounts    = np.random.poisson(total_reads_per_sample * (barcodeCopies/np.sum(barcodeCopies, axis=0)))

        countsDf = pd.DataFrame(data=readCounts, columns=sample_labels)
        if(outdir is not None):
            countsDf.to_csv(outdir+'counts.csv')
        # - - - - - -
        samplesDf = pd.DataFrame(index=range(num_samples))
        samplesDf['sample'] = sample_labels
        samplesDf['assay']  = [sample_label.split('-')[0] for sample_label in sample_labels]
        samplesDf['time']   = [int(sample_label.split('-')[1][1:]) for sample_label in sample_labels]
        samplesDf['bias_prevalence_true'] = bias_prev
        if(bias_prevalences is None):
            samplesDf['bias_trend_true']      = np.repeat(bias_prev_trends, len(times))
            samplesDf['bias_intercept_true']  = np.repeat(bias_prev_intercepts, len(times))
            samplesDf['bias_deviation_true']  = bias_prev_deviations
        if(outdir is not None):
            samplesDf.to_csv(outdir+'samples.csv')
        # - - - - - -
        variantsDf = pd.DataFrame(data=[i < num_variants_control for i in range(num_variants)], columns=['control_set'])
        variantsDf['bias_susceptibility_true'] = bias_susc
        if(outdir is not None):
            variantsDf.to_csv(outdir+'variants.csv')
        # - - - - - -
        fitnessesDf = pd.DataFrame(data=fitnesses, columns=assay_labels)
        if(outdir is not None):
            fitnessesDf.to_csv(outdir+'fitnesses_true.csv')

        dataset = [countsDf, samplesDf, variantsDf, fitnessesDf]

        return dataset if return_data else None


##############################################
##############################################
##############################################
##############################################

    
if __name__ == '__main__':
    # Running as a script
    print("-- bias_correction.py --")

    import argparse
    parser  = argparse.ArgumentParser()
    parser.add_argument("-c", "--counts", required=True)
    parser.add_argument("-s", "--samples", required=True)
    parser.add_argument("-v", "--variants", required=True)
    parser.add_argument("-cfg", "--config", required=True)
    parser.add_argument("-o", "--outdir", required=True)
    parser.add_argument("--num_iters", default=5, type=int)
    parser.add_argument("--output_intermediates", default=True, type=bool)
    parser.add_argument("--output_final", default=True, type=bool)
    parser.add_argument("--seed", default=1)
    args    = parser.parse_args()

    np.random.seed(args.seed)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set up bias correction object:
    print("[ Initializing. ]")
    debiaser = REBAR(counts=args.counts,
                     samples=args.samples,
                     variants=args.variants,
                     config=args.config,
                     outdir=args.outdir,
                     output_intermediates=args.output_intermediates,
                     output_final=args.output_final)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Perform bias correction:
    debiaser.run(num_iters=args.num_iters)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Save results to file
    # print("[ Saving outputs. ]")
    # debiaser.save(label='final', save_to_file=True)



           
            
















