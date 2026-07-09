# =============================================================================
# GMM Functions
# =============================================================================
# gmm_new()              – mixed-effects regression to derive GMM coefficients
# calculate_residuals()   – split total residuals into dBe, dS2S, dWSes components
# gmm_eval()              – scenario predictions and diagnostic plots
# Kothaetal2020Epe()      – ESM functional form (Kotha et al. 2020)
# gmpe_allM_YA15_Mh_Mref() – non-ESM functional form (YA15)
# multiplot()             – helper to arrange multiple ggplot objects on one page
# =============================================================================

# %||% became a base R builtin in R 4.4.0; the "reformulas" package (a lme4
# dependency) uses it unconditionally, which breaks lmer() summaries on
# older R versions. Polyfill it here if missing.
if (!exists("%||%", envir = globalenv())) {
  `%||%` <- function(x, y) if (is.null(x)) y else x
}


########## GMPE REGRESSION ############
# Fits a mixed-effects ground motion model via lmer (or rlmer) for each period/frequency.
#
# Arguments:
#   selected_data  – flatfile data.frame; must contain EQ_Code, StationCode, RJB, MAG,
#                    VS30, ev_depth_km, tHigh, fLow, fHigh, Mbin, Dist_flag, FM,
#                    and a column per IM named as returned by sprintf(fmt='X%#.3f').
#   periods        – character/numeric vector of periods (s) or frequencies (Hz);
#                    use "PGA" for peak ground acceleration (mapped to tt=0.01 s).
#   domain         – "SA" (spectral acceleration, g) or "FAS" (Fourier amplitude).
#   spes           – network identifier; "ESM" selects Kotha 2020 functional form,
#                    else (Japan) selects the YA15 functional form.
#   alg            – "lmer" for standard REML (lme4) or "rlmm" for robust fit (robustlmm).
#   min_no_event   – minimum number of records per event; events below this are dropped.
#   min_no_site    – minimum number of records per station (currently unused)
#   do_plot        – if TRUE, saves diagnostic JPEG figures to savefigto/.
#   savefigto      – directory path for output figures (must exist).
#
# Returns:
#   coeff_table data.frame with one row per period and columns for all GMM coefficients
#   and standard deviations (tau, phi0, phis2s, sigma).
#   writes coeff_table, full flatfile, dBe table, and dS2S table to CSV.
gmm_new <- function(selected_data, periods, domain, spes,
                     alg, min_no_event = 3, min_no_site = 3, do_plot = FALSE,
                     savefigto = "GMPE_figures") {
  
  if(spes == "ESM") {
    coeff_table <- data.frame("period" = periods,
                              "a" = c(0),
                              "b1" = c(0),
                              "Mref" = c(4.5),
                              "b2" = c(0),
                              "Mh" = c(5.7), 
                              "b3" = c(0),
                              "Rref" = c(30),
                              "c1" = c(0),
                              "h_D10" = c(4),
                              "h_10D20" = c(8),
                              "h_20D" = c(12),
                              "c2" = c(0),
                              "c3" = c(0),
                              "phis2s" = c(0),
                              "tau" = c(0),
                              "phi0" = c(0),
                              "sigma" = c(0),
                              "rock_adjustement" = c(0))
  
  } else {
    coeff_table <- data.frame("period" = periods,
                              "a" = c(0),
                              "b1" = c(0),
                              "Mref" = c(4.5),
                              "b2" = c(0),
                              "Mh" = c(0),
                              "b3" = c(0),
                              "c1" = c(0),
                              "Rs" = c(100),
                              "c2" = c(0),
                              "c3" = c(0),
                              "phis2s" = c(0),
                              "tau" = c(0),
                              "phi0" = c(0),
                              "sigma" = c(0),
                              "rock_adjustement" = c(0))
    
  }
  
  
  rock_data <- subset(selected_data, selected_data$VS30>760)
  selected_data$StationCode_Ch <- as.character(selected_data$StationCode)
  
  # Create table for ds2s and dbe
  ds2s_table0 <- data.frame("StationCode" = unique(selected_data$StationCode))
  ds2s_table0[['VS30']] <- selected_data[match(ds2s_table0$StationCode, selected_data$StationCode),"VS30"]

  dbe_table0 <- data.frame("EQ_Code" = unique(selected_data$EQ_Code))
  dbe_table0[['MAG']] <- selected_data[match(dbe_table0$EQ_Code, selected_data$EQ_Code),"MAG"]
  
  #t<-"0.01"
  for (t in periods) {
    print(t)
    
    IM <- ifelse(t == "PGA", t, sprintf(as.numeric(t), fmt = 'X%#.3f'))
    tt <- ifelse(t == "PGA", 0.01, as.numeric(t))
    
    # Create dataframe with only necessary columns;
    cols_of_interest = c('Address', 'EQ_Code', 'StationCode', 'RJB', 'MAG', 'VS30',
                         'ev_depth_km',
                         'tHigh','fLow','fHigh',
                         'Mbin', 'Dist_flag', 'FM', IM)
    ff_ss <- selected_data[cols_of_interest]
    
    ##### Define usable subset for the period ####
    if(domain == "FAS") {
      tu <- 'Hz'
      ff_ss <- subset(ff_ss, ((ff_ss[IM] > 0) & (ff_ss$fLow <= tt) & (ff_ss$fHigh >= tt)))
    } else {
      tu <- 's'
      ff_ss <- subset(ff_ss, ((ff_ss[IM] > 0) & (ff_ss$tHigh >= tt)))
    }
    
    ##### Phase 1 : Estimate c1 (geometric spreading), c3 (anelastic attenuation), and ds2s (site residuals) using all magnitude earthquakes #####
    ## Remove events with less than min_no_event records ####
    ff_ss <- ff_ss[which(ff_ss$EQ_Code %in% names(which(table(ff_ss$EQ_Code) >= min_no_event))),]
    
    print(paste("Number of rows for period", t, ":", nrow(ff_ss)))
    
    if(spes == "ESM") {
      ### For magnitude scaling ###
      Mref <- coeff_table[which(coeff_table$period == t),"Mref"]

      Mh <- coeff_table[which(coeff_table$period == t),"Mh"]

      ff_ss$Mref <- Mref

      ff_ss$Mh <- Mh

      ## Columns for coefficients b1, b2, b3 ##
      ff_ss$b1 <- ifelse(ff_ss$MAG <= ff_ss$Mh, ff_ss$MAG - ff_ss$Mh, 0)

      ff_ss$b2 <- ifelse(ff_ss$MAG <= ff_ss$Mh, (ff_ss$MAG - ff_ss$Mh)^2, 0)

      ff_ss$b3 <- ifelse(ff_ss$MAG <= ff_ss$Mh, 0, ff_ss$MAG - ff_ss$Mh)

      ### For distance scaling ###
      # ESM uses Rref (reference distance, fixed), not Rs (saturation distance used in YA15).
      # Stored as Rs here only so the plotting code below can use a single variable name.
      Rs <- coeff_table[which(coeff_table$period == t),"Rref"]
      ff_ss[["Rs"]] <- Rs
      
      Rref <- coeff_table[which(coeff_table$period == t),"Rref"]
      
      ff_ss$Rref <- Rref
      
      ## Depth bin for plots ##
      ff_ss$Dbin <- ifelse(ff_ss$ev_depth_km < 10,"D<10km",
                           ifelse(ff_ss$ev_depth_km < 20,"10km\u2264D<20km","20km\u2264D"))
      
      ff_ss$Dbin <- factor(ff_ss$Dbin, c("D<10km","10km\u2264D<20km","20km\u2264D"))
      
      ## Add depth dependent h parameter ##
      ff_ss$h <- ifelse(ff_ss$Dbin == "D<10km", coeff_table[which(coeff_table$period == t),"h_D10"],
                        ifelse(ff_ss$Dbin == "10km\u2264D<20km", coeff_table[which(coeff_table$period == t),"h_10D20"],
                               ifelse(ff_ss$Dbin == "20km\u2264D", coeff_table[which(coeff_table$period == t),"h_20D"],NA)))
      
      ## Add columns for c1 and c2 ##
      ff_ss$c1 <- log(sqrt(ff_ss$RJB^2+ff_ss$h^2)/sqrt(ff_ss$Rref^2+ff_ss$h^2))
      
      ff_ss$c2 <- (ff_ss$MAG - ff_ss$Mref)*log(sqrt(ff_ss$RJB^2+ff_ss$h^2)/sqrt(ff_ss$Rref^2+ff_ss$h^2))
      
      ## Add columns for c3 ##
      # With a scaling factor of 100 to estimate c3 well #
      ff_ss$c3 <- (sqrt(ff_ss$RJB^2+ff_ss$h^2) - sqrt(ff_ss$Rref^2+ff_ss$h^2))/100
    
    
    } else {
      
      #### c1 using only near source distances < 100km ####
      Rs <- coeff_table[which(coeff_table$period == t),"Rs"]
      ff_ss[["Rs"]] <- Rs

      ff_ss[["h"]] <- exp(2.303*(pmax((-0.05+0.15*ff_ss[["MAG"]]),(-1.72+0.43*ff_ss[["MAG"]]))))
      
      ## Columns for coefficients c1, c2, c3 ##
      ff_ss$c1 <- ifelse(ff_ss$RJB < ff_ss[["Rs"]], log(sqrt(ff_ss[["RJB"]]^2 + ff_ss[["h"]]^2)),log(sqrt(ff_ss[["Rs"]]^2 + ff_ss[["h"]]^2)))
      
      ff_ss$c2 <- ifelse(ff_ss$RJB >= ff_ss[["Rs"]], log(ff_ss[["RJB"]]/ff_ss[["Rs"]]),0)
      
      ff_ss$c3 <- ifelse(ff_ss$RJB >= ff_ss[["Rs"]], (ff_ss[["RJB"]] - ff_ss[["Rs"]]),0)
      
      
      ##### Magnitude scaling  #####
      if(domain == "FAS") {
        coeff_table[which(coeff_table$period == t),"Mh"] <- ifelse(tt > 10, 5.7, 5.5)#round(5.5 + 0.32*(log(tt)-log(0.1)),2))
      } else {
        coeff_table[which(coeff_table$period == t),"Mh"] <- ifelse(tt < 0.1, 5.5, 5.7)#round(5.5 + 0.32*(log(tt)-log(0.1)),2)
      }
      
      ff_ss[["Mh"]] <- coeff_table[which(coeff_table$period == t),"Mh"]
      
      ff_ss[["Mref"]] <- coeff_table[which(coeff_table$period == t),"Mref"]
      
      ## Columns for coefficients b1, b2, b3 ##
     ff_ss$b1 <- ifelse(ff_ss$MAG < ff_ss$Mref, (ff_ss$MAG - ff_ss$Mref),0)
  
      ff_ss$b2 <- ifelse(ff_ss$MAG > ff_ss$Mh, (ff_ss$Mh - ff_ss$Mref),
                         ifelse(ff_ss$MAG >= ff_ss$Mref,(ff_ss$MAG - ff_ss$Mref),0))
  
      ff_ss$b3 <- ifelse(ff_ss$MAG >= ff_ss$Mh, (ff_ss$MAG - ff_ss$Mh),0)
      
    }
    
    ff_ss[["IM_values"]] <- log(ff_ss[[IM]])
        
    lmer_fit0 <- lmer(IM_values ~ 1 + b1 + b2 + b3 + c1 + c2 + c3 +
                      (1|EQ_Code) + (1|StationCode),
                      data = ff_ss)
    print(summary(lmer_fit0))
    
    if(alg == "rlmm") {
      lmer_fit <- rlmer(IM_values ~ 1 + b1 + b2 + b3 + c1 + c2 + c3 +
                          (1|EQ_Code)  +(1|StationCode),
                        data = ff_ss, init = lmer_fit0)
      print(summary(lmer_fit))
    } else {
      lmer_fit <-  lmer_fit0
    }
      
    
    # Coeff tables # 
    coeff_table[which(coeff_table$period == t),
                c("a","b1","b2","b3","c1","c2","c3","phis2s","tau","phi0")] <- c(fixef(lmer_fit)[["(Intercept)"]],#0,
                                                                            fixef(lmer_fit)[["b1"]],
                                                                            fixef(lmer_fit)[["b2"]],fixef(lmer_fit)[["b3"]],
                                                                            fixef(lmer_fit)[["c1"]],fixef(lmer_fit)[["c2"]],fixef(lmer_fit)[["c3"]],
                                                                            attr(VarCorr(lmer_fit)$StationCode,"stddev")[["(Intercept)"]],
                                                                            attr(VarCorr(lmer_fit)$EQ_Code,"stddev")[["(Intercept)"]],
                                                                            attr(VarCorr(lmer_fit),"sc"))
    
    coeff_table[which(coeff_table$period == t),"sigma"] <- sqrt(coeff_table[which(coeff_table$period == t),"phis2s"]^2 +
                                                                  coeff_table[which(coeff_table$period == t),"tau"]^2 +
                                                                  coeff_table[which(coeff_table$period == t),"phi0"]^2)
    
    ### Make tables ###
    if(alg == "rlmm") {
      ## dbe ##
      # robustlmm does not expose posterior variances, so se_dbe/se_ds2s are unavailable.
      dbe_table <- data.frame("IM" = IM, "t" = tt,
                              "EQ_Code" = rownames(ranef(lmer_fit)$EQ_Code),
                              "wt"  = getME(lmer_fit, "w_b")$EQ_Code[["(Intercept)"]],
                              "dbe" = ranef(lmer_fit)$EQ_Code[["(Intercept)"]])
      ## ds2s ##
      ds2s_table <- data.frame("IM" = IM, "t" = tt,
                               "StationCode" = rownames(ranef(lmer_fit)$StationCode),
                               "wt"   = getME(lmer_fit, "w_b")$StationCode[["(Intercept)"]],
                               "ds2s" = ranef(lmer_fit)$StationCode[["(Intercept)"]])
      ## dwses ##
      dwses_table <- data.frame("IM" = IM, "t" = tt,
                                "record_id" = rownames(lmer_fit@frame),
                                "wt"    = getME(lmer_fit, "w_e"),
                                "dwses" = lmer_fit@resp$wtres)
    } else {
      ## dbe ##
      dbe_table <- data.frame("IM" = IM, "t" = tt,
                              "EQ_Code" = rownames(ranef(lmer_fit)$EQ_Code),
                              "dbe"    = ranef(lmer_fit)$EQ_Code[["(Intercept)"]],
                              "se_dbe" = sqrt(attr(ranef(lmer_fit, condVar = TRUE)$EQ_Code, "postVar")[1, 1, ]))
      ## ds2s ##
      ds2s_table <- data.frame("IM" = IM, "t" = tt,
                               "StationCode" = rownames(ranef(lmer_fit)$StationCode),
                               "ds2s"    = ranef(lmer_fit)$StationCode[["(Intercept)"]],
                               "se_ds2s" = sqrt(attr(ranef(lmer_fit, condVar = TRUE)$StationCode, "postVar")[1, 1, ]))
      
      ## dwses ##
      dwses_table <- data.frame("IM" = IM, "t" = tt,
                                "record_id" = rownames(lmer_fit@frame),
                                "address" = paste(as.character(lmer_fit@frame$EQ_Code),lmer_fit@frame$StationCode, sep = "_"),
                                #"wt" = getME(lmer_fit,"w_e"),
                                "dwses" = lmer_fit@resp$wtres)
    }
      
    # Save the tables #
    ## Add dbe to the flatfile ##
    dbe_table0[[paste("dbe_",t,sep = "")]] <- dbe_table[match(dbe_table0$EQ_Code, dbe_table$EQ_Code),"dbe"]
    ff_ss[[paste("dbe_",IM,sep = "")]] <- dbe_table[match(ff_ss$EQ_Code, dbe_table$EQ_Code),"dbe"]
    ff_ss[[paste("se_dbe_",IM,sep = "")]] <- dbe_table[match(ff_ss$EQ_Code, dbe_table$EQ_Code),"se_dbe"]
    selected_data[[paste("dbe_",IM,sep = "")]] <- dbe_table[match(selected_data$EQ_Code, dbe_table$EQ_Code),"dbe"]
    selected_data[[paste("se_dbe_",IM,sep = "")]] <- dbe_table[match(selected_data$EQ_Code, dbe_table$EQ_Code),"se_dbe"]
    
    ## Add ds2s to the flatfile ##
    ds2s_table0[[paste("ds2s_",t,sep = "")]] <- ds2s_table[match(ds2s_table0$StationCode, ds2s_table$StationCode),"ds2s"]
    ff_ss[[paste("ds2s_",IM,sep = "")]] <- ds2s_table[match(ff_ss$StationCode, ds2s_table$StationCode),"ds2s"]
    ff_ss[[paste("se_ds2s_",IM,sep = "")]] <- ds2s_table[match(ff_ss$StationCode, ds2s_table$StationCode),"se_ds2s"]
    selected_data[[paste("ds2s_",IM,sep = "")]] <- ds2s_table[match(selected_data$StationCode, ds2s_table$StationCode),"ds2s"]
    selected_data[[paste("se_ds2s_",IM,sep = "")]] <- ds2s_table[match(selected_data$StationCode, ds2s_table$StationCode),"se_ds2s"]
    
    rock_data[[paste("ds2s_",IM,sep = "")]] <- ds2s_table[match(rock_data$StationCode, ds2s_table$StationCode),"ds2s"]
    coeff_table[which(coeff_table$period == t),"rock_adjustement"] <- mean(rock_data[!duplicated(rock_data$StationCode),paste("ds2s_",IM,sep = "")], na.rm = TRUE)
    
    ## Add dwses to the flatfile ##
    ff_ss[[paste("dwses_",IM,sep = "")]] <- dwses_table[match(rownames(ff_ss), dwses_table$record_id),"dwses"]
    selected_data[[paste("dwses_",IM,sep = "")]] <- dwses_table[match(rownames(selected_data), dwses_table$record_id),"dwses"]
     
    ### Adjust the observed ground motions ##
    if(spes == "ESM") {
      ff_ss[[paste("mag_dist_adj_",IM,sep="")]] <- exp(log(ff_ss[[IM]])- log(Kothaetal2020Epe(t,ff_ss$MAG,ff_ss$RJB,ff_ss$ev_depth_km,coeff_table,domain)))
      selected_data[[paste("mag_dist_adj_",IM,sep="")]] <- exp(log(selected_data[[IM]])- log(Kothaetal2020Epe(t,selected_data$MAG,selected_data$RJB,selected_data$ev_depth_km,coeff_table,domain)))
      #selected_data[[paste("mag_dist_adj_rock_",IM,sep="")]] <- exp(log(selected_data[[IM]])- log(Kothaetal2020Epe(t,selected_data$MAG,selected_data$RJB,selected_data$ev_depth_km,coeff_table,domain) + coeff_table[which(coeff_table$period == t),"rock_adjustement"]))
    } else {
      ff_ss[[paste("mag_dist_adj_",IM,sep="")]] <- exp(log(ff_ss[[IM]])- log(gmpe_allM_YA15_Mh_Mref(t,ff_ss$MAG,ff_ss$RJB,coeff_table)))
      selected_data[[paste("mag_dist_adj_",IM,sep="")]] <- exp(log(selected_data[[IM]])- log(gmpe_allM_YA15_Mh_Mref(t,selected_data$MAG,selected_data$RJB,coeff_table)))
      #selected_data[[paste("mag_dist_adj_rock_",IM,sep="")]] <- exp(log(selected_data[[IM]])- log(gmpe_allM_YA15_Mh_Mref(t,selected_data$MAG,selected_data$RJB,coeff_table) + coeff_table[which(coeff_table$period == t),"rock_adjustement"]))
    }
    
    if (do_plot){
      ## Distance scaling plot ##
      dist_scaling_plot <- ggplot(ff_ss[order(ff_ss$Mbin),],aes(x = RJB, color = Mbin))+
        annotate("text",label = paste("T = ",t,tu,sep = ""), x = 300, y = max(ff_ss[[IM]]/2), size = 7.5, family = "Cambria")+
        geom_point(aes_string(y = IM),shape = 1)+
        scale_x_log10(expression(R[JB]~"(km)"), limits = c(0.5,600), labels = comma )+
        scale_y_log10(paste(domain," (g)",sep = ""), labels = comma)+
        geom_vline(xintercept = Rs, color = "blue") +
        scale_color_colorblind("")+
        theme_bw()+
        guides(colour = guide_legend(override.aes = list(shape = 15, size = 6)))+
        theme(text = element_text(size = 20, family = "Cambria"),
              legend.title = element_blank(),
              legend.position = c(0.2,0.2),
              legend.direction = "vertical",
              legend.key.width = unit(1,"cm"))+
        annotation_logticks(sides = "bl")
      
      ## Distance scaled plot ##
      distance_scaled_plot <- ggplot(ff_ss[order(ff_ss$Dist_flag),],aes(x = RJB, color = Dist_flag))+
        geom_point(aes_string(y = paste("mag_dist_adj_",IM,sep="")),shape = 1)+
        facet_wrap(~Mbin, ncol = 1)+
        geom_vline(xintercept = Rs, color = "blue") +
        scale_x_log10(expression(R[JB]~"(km)"), limits = c(0.5,600))+
        scale_y_log10(expression(domain[R[ref]]~(g)),
                      limits = c(min(ff_ss[,paste("mag_dist_adj_",IM,sep = "")]),max(ff_ss[,paste("mag_dist_adj_",IM,sep = "")])), labels = comma)+
        scale_color_colorblind(expression(M[W]~"bin"))+
        theme_bw()+
        guides(colour = guide_legend(override.aes = list(shape = 15, size = 6)))+
        theme(text = element_text(size = 20, family = "Cambria"),
              legend.title = element_blank(),
              strip.background = element_blank(),
              legend.position = "none",
              legend.direction = "vertical",
              legend.key.width = unit(1,"cm"))+
        annotation_logticks(sides = "bl")

      ## Magnitude scaling plot  ##
      mag_scaling_plot <- ggplot(ff_ss[!duplicated(ff_ss$EQ_Code),], aes(x = MAG, color = FM))+
        geom_point(aes_string(y = paste("mag_dist_adj_",IM,sep = "")),shape = 1)+
        stat_summary_bin(aes_string(y = paste("mag_dist_adj_",IM,sep = "")),
                         fun="mean", bins=10,
                         color='red', size=1, geom='point')+
        stat_summary_bin(aes_string(y = paste("mag_dist_adj_",IM,sep = "")),
                         fun.data="mean_se", bins=10,
                         color='red', size =1,  geom='errorbar')+
        geom_vline(xintercept =   coeff_table[which(coeff_table$period == t),"Mref"], color = "darkgreen") +
        geom_vline(xintercept =   coeff_table[which(coeff_table$period == t),"Mh"], color = "blue") +
        scale_x_continuous(expression(M[W]), limits = c(3.25,8))+
        #scale_color_brewer("Focal mechanism",palette="Set1")+
        scale_y_log10(expression(bquote(.(domain)[R[ref]]~(g))), limits = c(min(ff_ss[!duplicated(ff_ss$EQ_Code),paste("mag_dist_adj_",IM,sep = "")]),
                                                                 max(ff_ss[!duplicated(ff_ss$EQ_Code),paste("mag_dist_adj_",IM,sep = "")])), labels = comma)+
        guides(colour = guide_legend(override.aes = list(shape = 15, size = 6)))+
        theme_bw()+
        theme(text = element_text(size = 20, family = "Cambria"),
              legend.title = element_blank(),
              legend.position = c(0.75,0.15),
              legend.direction = "vertical",
              legend.key.width = unit(1,"cm"))+
        annotation_logticks(sides = "l")

      ## Magnitude and distance scaled plot ##'
      mag_dist_scaled_plot <- ggplot(ff_ss[order(ff_ss$FM),],aes(x = RJB, color = FM))+
        geom_point(aes_string(y = paste("mag_dist_adj_",IM,sep="")),shape = 1)+
        geom_vline(xintercept = Rs, color = "blue") +
        facet_wrap(~Mbin, ncol = 1)+
        scale_x_log10(expression(R[JB]~"(km)"), limits = c(0.5,600))+
        scale_y_log10(expression(bquote(.(domain)[R[ref]]~(g))),
                      limits = c(min(ff_ss[,paste("mag_dist_adj_",IM,sep = "")]),
                                 max(ff_ss[,paste("mag_dist_adj_",IM,sep = "")])), labels = comma)+
        #scale_color_brewer(palette="Set1")+
        theme_bw()+
        guides(colour = guide_legend(override.aes = list(shape = 15, size = 6)))+
        theme(text = element_text(size = 20, family = "Cambria"),
              legend.title = element_blank(),
              legend.position = "none",
              strip.background = element_blank(),
              legend.direction = "vertical",
              legend.key.width = unit(1,"cm"))+
        annotation_logticks(sides = "bl")
    }
    
    ## Add mean of event observations,site-corrected,event-corrected, event-and-site-corrected, IM values to the flatfile ##
    selected_data[,paste("site_corrected_",IM,sep="")] <- exp(log(selected_data[,IM]) - selected_data[,paste("ds2s_",IM,sep="")])
    selected_data[,paste("event_corrected_",IM,sep="")] <- exp(log(selected_data[,IM]) - selected_data[,paste("dbe_",IM,sep="")])
    selected_data[,paste("event_and_site_corrected_",IM,sep="")] <-exp(log(selected_data[,IM]) - selected_data[,paste("dbe_",IM,sep="")] - selected_data[,paste("ds2s_",IM,sep="")])

    
    if (do_plot){
      #### Residual analysis plots ###
      ## dbe vs Magnitude plot  ##
      dbe_M_plot <- ggplot(ff_ss[!duplicated(ff_ss$EQ_Code),], aes(x = MAG))+
        annotate("text", x = 3.875, y = 1.75,label = paste("T = ",t,tu,sep = ""), size = 5, family = "Cambria")+
        # geom_point(data = subset(ff_ss[!duplicated(ff_ss$EQ_Code),],Dist_flag == "Near- and far-source"),
        #            aes_string(y = paste("dbe_",IM,sep = ""), color = "EQ.Depth.3..km."), shape = 1)+
        geom_point(data = subset(ff_ss[!duplicated(ff_ss$EQ_Code),],Dist_flag == "Near- and far-source"),
                   aes_string(y = paste("dbe_",IM,sep = "")), color = "black", shape = 1)+
        stat_summary_bin(aes_string(y = paste("dbe_",IM,sep = "")),
                         fun="mean", bins=10,
                         color='red', size=1, geom='point')+
        stat_summary_bin(aes_string(y = paste("dbe_",IM,sep = "")),
                         fun.data="mean_cl_normal", bins=10,
                         color='red',  geom='errorbar')+
        scale_shape(solid = FALSE)+
        # scale_x_continuous(expression(M[W]), limits = c(3.25,8.5))+
        scale_x_continuous(expression(M[W]), limits = c(3.25,7.5))+
        scale_color_gradientn("Depth(Km)", colours = rev(rainbow(9)))+
        # guides(colour = guide_legend(override.aes = list(shape = 15, size = 6)))+
        scale_y_continuous(expression(delta*B[e]), limits = c(-2,2))+
        theme_bw()+
        theme(text = element_text(size = 15, family = "Cambria"),
              legend.position = c(0.875,0.5),
              legend.background = element_rect(),
              legend.direction = "vertical",
              legend.key.height = unit(0.75,"cm"))
      
      ## dbs vs VS30 plot  ##
      ds2s_VS30_plot <-   ggplot(ff_ss[!duplicated(ff_ss$StationCode),], aes(x = VS30))+
        # annotate("text", x = 1600, y = 1.75,label = paste("T = ",t,tu,sep = ""), size = 5)+
        # geom_point(aes_string(y = paste("ds2s_",IM,sep = ""), color = "h800..m."), shape = 1)+
        geom_point(aes_string(y = paste("ds2s_",IM,sep = "")), color = "black", shape = 1)+
        geom_smooth(aes_string(y = paste("ds2s_",IM,sep = "")), color = "red", method = "loess", span = 1)+
        scale_x_log10(expression(V[s30]~(m/s)), breaks = c(180,360,760,1500), limits = c(100,2000))+
        scale_y_continuous(expression(delta*S2S[s]), limits = c(-2,2))+
        scale_color_gradientn(expression(H[800]*(m)), colours = rev(rainbow(9)), trans = "log", breaks = c(1,10,100,500))+
        theme_bw()+
        theme(text = element_text(size = 15, family = "Cambria"),
              legend.position = c(0.9,0.5),
              legend.direction = "vertical",
              legend.key.height = unit(0.75,"cm"))+
        annotation_logticks(sides = "b")
      
      ## dwses vs Distance plot  ##
      dwses_RJB_plot <- ggplot(ff_ss,aes(x = RJB))+
        geom_point(aes_string(y = paste("dwses_",IM,sep = "")), color = "black", shape = 1)+
        stat_summary_bin(aes_string(y = paste("dwses_",IM,sep = "")),
                         fun="mean", bins=10,
                         color='red', size=1, geom='point')+
        stat_summary_bin(aes_string(y = paste("dwses_",IM,sep = "")),
                         fun.data="mean_cl_normal", bins=10,
                         color='red', geom='errorbar')+
        scale_color_gradientn(colours = rev(rainbow(7)))+
        scale_x_log10(expression(R[JB]~(km)), limits = c(0.5,600))+
        scale_y_continuous(expression(delta*W*S[es]), limits = c(-3,3))+
        theme_bw()+
        theme(text = element_text(size = 15, family = "Cambria"),
              legend.title = element_blank(),
              legend.position = "none",
              legend.direction = "vertical",
              legend.key.width = unit(1,"cm"))+
        annotation_logticks(sides = "b")
    }
    
    if (do_plot){
      #### Save plots ####
      jpeg(file = paste(savefigto,"/non_par_",t,"_",spes,".jpeg",sep=""),
           width = 18, height = 6, units = 'in', res = 300)
      
      multiplot(dist_scaling_plot, mag_scaling_plot, mag_dist_scaled_plot, cols = 3)
      
      dev.off()
      
      jpeg(file = paste(savefigto,"/res_",t,"_",spes,".jpeg",sep=""),
           width = 6, height = 9, units = 'in', res = 300)

      multiplot(dbe_M_plot,ds2s_VS30_plot,dwses_RJB_plot)

      dev.off()

      rm(ff_ss,IM,t,Rs,distance_scaled_plot,
         dist_scaling_plot, mag_scaling_plot,mag_dist_scaled_plot,
         dbe_M_plot,dwses_RJB_plot,ds2s_VS30_plot)

    }
  }
  
  #### Write CSV files for the flatfile and coefficient table ####
  coeff_table_allM_YA15_Mh_Mref <- coeff_table
  write.csv(coeff_table, paste("coeff_table_",spes,domain,alg,".csv", sep=""))
  write.csv(selected_data, paste("gmm_data_",spes,domain,alg,".csv", sep=""))
  
  write.csv(dbe_table0, paste("dbe_",spes ,domain,alg,".csv", sep=""))
  write.csv(ds2s_table0, paste("dS2S_",spes ,domain,alg,".csv", sep=""))
  
  return(coeff_table)  
}


########## GMPE FUNCTION ############
# Predicts median ground motion using the YA15-style functional form
# (bilinear magnitude scaling with Mref/Mh hinges; two-segment distance decay with Rs).
# Used for Japan networks (e.g. KiK-net or K-net).
#
# Arguments:
#   Period       – scalar period identifier matching a row in coeff_table$period.
#   M            – vector of moment magnitudes.
#   Distance     – vector of Joyner-Boore distances (km).
#   coeff_table  – data.frame of fitted coefficients (output of gmpe_new()).
#
# Returns: numeric vector of median IM predictions (same units as the fitted data).
gmpe_allM_YA15_Mh_Mref <- function(Period, M, Distance, coeff_table) {
  
  Period <- Period
  M <- M
  Distance <- Distance
  coeff_table <- coeff_table
  
  a <- coeff_table[which(coeff_table$period == Period),"a"]
  
  b1 <- coeff_table[which(coeff_table$period == Period),"b1"]
  Mref <- coeff_table[which(coeff_table$period == Period),"Mref"]
  b2 <- coeff_table[which(coeff_table$period == Period),"b2"]
  Mh <- coeff_table[which(coeff_table$period == Period),"Mh"]
  b3 <- coeff_table[which(coeff_table$period == Period),"b3"]
  
  c1 <- coeff_table[which(coeff_table$period == Period),"c1"]
  Rs <- coeff_table[which(coeff_table$period == Period),"Rs"]
  
  c2 <- coeff_table[which(coeff_table$period == Period),"c2"]
  c3 <- coeff_table[which(coeff_table$period == Period),"c3"]
  
  #### Magnitude scaling ####
  exprM1 <- M - Mref
  
  exprM2 <- b1 * exprM1
  
  exprM3 <- b2 * exprM1
  
  exprM4 <- M - Mh
  
  exprM5 <- b2 * (Mh - Mref) + b3 * exprM4
  
  FM <-  ifelse(M < Mref, exprM2,
                ifelse(M < Mh, exprM3, exprM5))
  
  #### Distance scaling ####
  h <- exp(2.303*(pmax((-0.05+0.15*M),(-1.72+0.43*M))))
  #h <- 10^pmax((-0.05+0.15*M),(-1.72+0.43*M))
  
  Rh <- sqrt(Distance^2 + h^2)
  
  expr_R1 <-  c1 * log(Rh)
  
  expr_R2 <- c1 * log(sqrt(Rs^2+h^2)) + c2 * log(Distance/Rs) + c3 * (Distance - Rs)
  
  FD <- ifelse(Distance < Rs, expr_R1,expr_R2)
  
  #### Prediction ####
  value <- exp(a + FM + FD)
  
  value
}



# Predicts median ground motion using the Kotha et al. (2020) ESM functional form.
# Magnitude scaling: quadratic below Mh, linear above.
# Distance scaling: geometrical spreading with magnitude-dependent c2 term; anelastic c3.
# Depth-dependent pseudo-depth h (three bins: D<10, 10≤D<20, D≥20 km).
#
# Arguments:
#   Period       – scalar period identifier matching a row in coeff_table$period.
#   M            – vector of moment magnitudes.
#   R            – vector of Joyner-Boore distances (km).
#   D            – vector of hypocentral depths (km).
#   coeff_table  – data.frame of fitted coefficients (output of gmpe_new()).
#   domain       – "SA" or "FAS"; controls whether c2 distance term is magnitude-dependent.
#
# Returns: numeric vector of median IM predictions.
Kothaetal2020Epe <- function(Period, M, R, D, coeff_table, domain) {
    
    Period <- Period
    M <- M
    R <- R
    D <- D
    coeff_table <- coeff_table
    
    if ("a" %in% colnames(coeff_table)) {
      a <- coeff_table[which(coeff_table$period == Period),"a"]
    } else {
      a <- coeff_table[which(coeff_table$period == Period),"e1"]
    }
    
    b1 <- coeff_table[which(coeff_table$period == Period),"b1"]
    Mref <- coeff_table[which(coeff_table$period == Period),"Mref"]
    b2 <- coeff_table[which(coeff_table$period == Period),"b2"]
    Mh <- coeff_table[which(coeff_table$period == Period),"Mh"]
    b3 <- coeff_table[which(coeff_table$period == Period),"b3"]
    
    c1 <- coeff_table[which(coeff_table$period == Period),"c1"]
    
    c2 <- coeff_table[which(coeff_table$period == Period),"c2"]
    c3 <- coeff_table[which(coeff_table$period == Period),"c3"]
    
    Rref <- coeff_table[which(coeff_table$period == Period),"Rref"]
    
    h_D10 <- coeff_table[which(coeff_table$period == Period),"h_D10"]
    h_10D20 <- coeff_table[which(coeff_table$period == Period),"h_10D20"]
    h_D20 <- coeff_table[which(coeff_table$period == Period),"h_20D"]
    
    h <-ifelse(D < 10, h_D10,
               ifelse(((D >= 10) & (D < 20)), h_10D20, h_D20))
    
    ## FM ##
    Mh <- Mh 
    exprM1 <- M - Mh
    exprM2 <- exprM1 ^ 2
    exprM3 <- b1 * exprM1 + b2 * exprM2
    exprM4 <- b3 * exprM1
    valueFM <- ifelse(M<=Mh, exprM3, exprM4)
    
    ##FD ##
    Mref <- Mref
    Rref <- Rref
    h <- h
    exprD1 <- M - Mref 
    exprD2 <- c1 + c2 * exprD1  # [c1 + c2(M-Mref)]
    exprD3 <- R^2 + h^2  # [Rjb^2 + h^2]
    exprD4 <- sqrt(exprD3)  # [sqrt[Rjb^2 + h^2]]
    exprD5 <- exprD4/sqrt(Rref^2 + h^2)      # [sqrt[Rjb^2 + h^2]/sqrt[Rref^2 + h^2]]
    exprD6 <- log(exprD5)   # LN[sqrt[Rjb^2 + h^2]/sqrt[Rref^2 + h^2]]
    exprD7 <- (exprD4 - sqrt(Rref^2 + h^2))/100  # [sqrt[Rjb^2 + h^2] - sqrt[Rref^2 + h^2]]/100
    
    if(domain == "FAS") {
      valueFD <- c1 * exprD6 + c3 * exprD7
    } else {
      valueFD <- exprD2 * exprD6 + c3 * exprD7
    }
    
    ## Value ##
    value <- exp(a + valueFD + valueFM) 
    
    value
    
  }



########## SPLIT RESIDUALS ############
# Computes total residuals relative to a reference GMM and splits them (using lmer/rlmer) into
# between-event (dBe), site-to-site (dS2S), and within-event (dWSes) components
#
# Arguments:
#   gmpe_res     – flatfile data.frame (same columns as selected_data in gmpe_new()).
#                  Despite the name, this is the *input* flatfile, not pre-computed residuals.
#   periods      – vector of periods/frequencies, same as passed to gmpe_new().
#   gmpe         – string label for the reference model; controls which prediction
#                  function is called:
#                    "Gmm"     – uses the fitted coeff_table (ESM → Kothaetal2020Epe,
#                                other/Japan → gmpe_allM_YA15_Mh_Mref)
#                    "Kotha18" – gmpe_allM_YA15_Mh_Mref with coeff_table (Kotha et al., 2018)
#                    "BSSA14"  – external BSSA14() function (Boore et al., 2014; must be loaded separately)
#   coeff_table  – coefficient data.frame from gmpe_new().
#   domain       – "SA" or "FAS".
#   spes         – network identifier ("ESM" or other/Japan).
#   alg          – "lmer" or "rlmm".
#   min_no_event – minimum number of records per event for the residual-splitting.
#   min_no_site  – minimum number of records per station for the residual-splitting (currently unused).
#
# Returns: gmpe_res data.frame augmented with GMPE predictions, total residuals,
#          dBe, dS2S, dWSes, and dwe (= eps - dBe) columns per period.
#   writes augmented flatfile and dBe/dS2S tables to CSV.
calculate_residuals <- function(gmpe_res, periods, gmpe, coeff_table, domain, spes,
                           alg, min_no_event = 3, min_no_site = 3,
                           do_plot = FALSE, savefigto = "residual_plots") {
  
  if (gmpe=="Kotha18"){
    rock_adj <- read.csv('ref_cluster_8_mean_ds2s.csv')
  }
  
  gmpe_res$StationCode_Ch <- as.character(gmpe_res$StationCode)
  
  # Create table for ds2s and dbe
  ds2s_table0 <- data.frame("StationCode" = unique(gmpe_res$StationCode))
  ds2s_table0[['VS30']] <- gmpe_res[match(ds2s_table0$StationCode, gmpe_res$StationCode),"VS30"]
  
  dbe_table0 <- data.frame("EQ_Code" = unique(gmpe_res$EQ_Code))
  dbe_table0[['MAG']] <- gmpe_res[match(dbe_table0$EQ_Code, gmpe_res$EQ_Code),"MAG"]
  
  for (t in periods) {
    print(t)
    
    IM <- ifelse(t == "PGA", t, sprintf(as.numeric(t), fmt = 'X%#.3f'))
    tt <- ifelse(t == "PGA", 0.01, as.numeric(t))
    if (gmpe=="Kotha18"){
      # Extract rock adjustment:
      adj <- ifelse(tt<=2, rock_adj[which(rock_adj$t == tt),"adjustment"], 0)
      
      #### Predict the ground motion ####
      pred <- mapply(gmpe_allM_YA15_Mh_Mref,tt,gmpe_res$MAG, gmpe_res$RJB)
      
      gmpe_res[,paste("GMPE_",t,sep = "")] <- pred
      gmpe_res[,paste("GMPE_rock_",t,sep = "")] <- exp(log(pred) + adj)
      gmpe_res[,paste("Sigma_",t,sep = "")] <- sqrt(coeff_table[which(coeff_table$period == tt),"phis2s"]^2 +
                                                      coeff_table[which(coeff_table$period == tt),"tau"]^2 +
                                                      coeff_table[which(coeff_table$period == tt),"phi0"]^2)
      
    } else if (gmpe=="BSSA14"){
      
      # #### Predict the ground motion BSSA14 ####
      pred = BSSA14(tt,gmpe_res$MAG, gmpe_res$FM, gmpe_res$RJB, gmpe_res$VS30,-1,'Japan')
      gmpe_res[,paste("GMPE_",t,sep = "")] <- pred["Mean"]
      gmpe_res[,paste("Sigma_",t,sep = "")] <- pred["Sigma"]
      
      pred_rock = BSSA14_PGAr(tt,gmpe_res$MAG, gmpe_res$FM, gmpe_res$RJB)
      gmpe_res[,paste("GMPE_rock_",t,sep = "")] <- pred_rock["Mean"]
      gmpe_res[,paste("Sigma_rock_",t,sep = "")] <- pred_rock["Sigma"]
      
      
    } else {
      if (spes=="ESM"){
        # Extract rock adjustment:
        adj <- coeff_table[which(coeff_table$period == t),"rock_adjustement"]
        
        #### Predict the ground motion ####
        pred <- Kothaetal2020Epe(t,gmpe_res$MAG, gmpe_res$RJB, gmpe_res$ev_depth_km, coeff_table,domain)
        
        gmpe_res[,paste("GMPE_",t,sep = "")] <- pred
        gmpe_res[,paste("GMPE_rock_",t,sep = "")] <- exp(log(pred) + adj)
        gmpe_res[,paste("Sigma_",t,sep = "")] <- sqrt(coeff_table[which(coeff_table$period == t),"phis2s"]^2 +
                                                        coeff_table[which(coeff_table$period == t),"tau"]^2 +
                                                        coeff_table[which(coeff_table$period == t),"phi0"]^2)
      } else {
                                                      
        # Extract rock adjustment:
        adj <- coeff_table[which(coeff_table$period == t),"rock_adjustement"]
        
        #### Predict the ground motion ####
        pred <- gmpe_allM_YA15_Mh_Mref(t,gmpe_res$MAG, gmpe_res$RJB,coeff_table)
        
        gmpe_res[,paste("GMPE_",t,sep = "")] <- pred
        gmpe_res[,paste("GMPE_rock_",t,sep = "")] <- exp(log(pred) + adj)
        gmpe_res[,paste("Sigma_",t,sep = "")] <- sqrt(coeff_table[which(coeff_table$period == t),"phis2s"]^2 +
                                                        coeff_table[which(coeff_table$period == t),"tau"]^2 +
                                                        coeff_table[which(coeff_table$period == t),"phi0"]^2)
    }}
    
    #### Residuals after removing site response ####
    gmpe_res[,paste("eps_full_",t,sep = "")] <- log(gmpe_res[IM]) - log(gmpe_res[,paste("GMPE_",t,sep = "")])
    gmpe_res[,paste("eps_",t,sep = "")] <- log(gmpe_res[IM]) - log(gmpe_res[,paste("GMPE_rock_",t,sep = "")])
    
    
    #### Lmer to split the eps in to site residuals ####
    ##### Define usable subset for the period ####
    if(domain == "FAS") {
      tu <- 'Hz'
      gmpe_res_ss <- subset(gmpe_res, ((gmpe_res[IM] > 0) & (gmpe_res$fLow <= tt) & (gmpe_res$fHigh >= tt)))
    } else {
      tu <- 's'
      gmpe_res_ss <- subset(gmpe_res, ((gmpe_res[IM] > 0) & (gmpe_res$tHigh >= tt)))
    }

    
    ## Remove events with less than 3 records ####
    gmpe_res_ss <- gmpe_res_ss[which(gmpe_res_ss$EQ_Code %in% names(which(table(gmpe_res_ss$EQ_Code) >= min_no_event))),]

    gmpe_res_ss <- subset(gmpe_res_ss, !is.na(gmpe_res_ss[,paste("eps_",t,sep = "")]))
    print(paste('Number of usable records for period', tt, ':', nrow(gmpe_res_ss)))
    
    gmpe_res_ss['res'] <- gmpe_res_ss[,paste("eps_",t,sep = "")]
    
    # Create dataframe with only necessary columns;
    cols_of_interest = c('Address', 'EQ_Code', 'StationCode', 'res')
    gmpe_res_rr <- gmpe_res_ss[cols_of_interest]
    
    lmer_fit0 <- lmer(res ~ 1 + (1|StationCode) + (1|EQ_Code), data = gmpe_res_rr)
    print(summary(lmer_fit0))
    
    if(alg == "rlmm") {
      lmer_fit <- rlmer(res ~ 1 + (1|StationCode) + (1|EQ_Code), data = gmpe_res_rr, init = lmer_fit0)
      print(summary(lmer_fit))

    } else {
      lmer_fit <-  lmer_fit0
    }
    
    
    ### Make tables ###
    
    if(alg == "rlmm") {
      ## dbe ##
      dbe_table <- data.frame("IM" = IM, "t" = tt,
                              "EQ_Code" = rownames(ranef(lmer_fit)$EQ_Code),
                              "wt" = getME(lmer_fit,"w_b")$EQ_Code[["(Intercept)"]],
                              "dbe" = ranef(lmer_fit)$EQ_Code[["(Intercept)"]])
      ## ds2s ##
      ds2s_table <- data.frame("IM" = IM, "t" = tt,
                               "StationCode" = rownames(ranef(lmer_fit)$StationCode),
                               "wt" = getME(lmer_fit,"w_b")$StationCode[["(Intercept)"]],
                               "ds2s" = ranef(lmer_fit)$StationCode[["(Intercept)"]])
      ## dwses ##
      dwses_table <- data.frame("IM" = IM, "t" = tt,
                                "record_id" = rownames(lmer_fit@frame),
                                "wt" = getME(lmer_fit,"w_e"),
                                "dwses" = lmer_fit@resp$wtres)
    } else {
      ## dbe ##
      dbe_table <- data.frame("IM" = IM, "t" = tt,
                              "EQ_Code" = rownames(ranef(lmer_fit)$EQ_Code),
                              "dbe"    = ranef(lmer_fit)$EQ_Code[["(Intercept)"]],
                              "se_dbe" = sqrt(attr(ranef(lmer_fit, condVar = TRUE)$EQ_Code, "postVar")[1, 1, ]))
      ## ds2s ##
      ds2s_table <- data.frame("IM" = IM, "t" = tt,
                               "StationCode" = rownames(ranef(lmer_fit)$StationCode),
                               "ds2s"    = ranef(lmer_fit)$StationCode[["(Intercept)"]],
                               "se_ds2s" = sqrt(attr(ranef(lmer_fit, condVar = TRUE)$StationCode, "postVar")[1, 1, ]))
      
      ## dwses ##
      dwses_table <- data.frame("IM" = IM, "t" = tt,
                                "record_id" = rownames(lmer_fit@frame),
                                "dwses" = lmer_fit@resp$wtres)
    }
    
    
    ds2s_table0[[paste("ds2s_",t,sep = "")]] <- ds2s_table[match(ds2s_table0$StationCode, ds2s_table$StationCode),"ds2s"]
    gmpe_res[[paste("ds2s_",t,sep = "")]] <- ds2s_table[match(gmpe_res$StationCode, ds2s_table$StationCode),"ds2s"]
    
    dbe_table0[[paste("dbe_",t,sep = "")]] <-  dbe_table[match(dbe_table0$EQ_Code, dbe_table$EQ_Code),"dbe"]
    gmpe_res[[paste("dbe_",t,sep = "")]] <- dbe_table[match(gmpe_res$EQ_Code, dbe_table$EQ_Code),"dbe"]
    
    # Assign within-event residuals by record_id (original row names from gmpe_res),
    # which lmer preserves in @frame regardless of internal grouping order.
    gmpe_res[dwses_table$record_id, paste("dwses_", t, sep = "")] <- dwses_table$dwses
    
    gmpe_res[,paste("dwe_",t,sep = "")] <- gmpe_res[,paste("eps_",t,sep = "")] -gmpe_res[,paste("dbe_",t,sep="")]

    ## Observed IM with the between-event and site terms removed, for use in gmpe_eval() ##
    gmpe_res[,paste("event_and_site_corrected_",t,sep = "")] <- exp(log(gmpe_res[,IM]) -
                                                                      gmpe_res[,paste("dbe_",t,sep = "")] -
                                                                      gmpe_res[,paste("ds2s_",t,sep = "")])

    if (do_plot) {
      ## dbe vs Magnitude ##
      dbe_M_plot <- ggplot(gmpe_res[!duplicated(gmpe_res$EQ_Code),], aes(x = MAG)) +
        annotate("text", x = 3.875, y = 1.75, label = paste("T = ", t, tu, sep = ""), size = 5, family = "Cambria") +
        geom_point(data = subset(gmpe_res[!duplicated(gmpe_res$EQ_Code),], Dist_flag == "Near- and far-source"),
                   aes_string(y = paste("dbe_", t, sep = "")), color = "black", shape = 1) +
        stat_summary_bin(aes_string(y = paste("dbe_", t, sep = "")),
                         fun = "mean", bins = 10, color = 'red', size = 1, geom = 'point') +
        stat_summary_bin(aes_string(y = paste("dbe_", t, sep = "")),
                         fun.data = "mean_cl_normal", bins = 10, color = 'red', geom = 'errorbar') +
        scale_x_continuous(expression(M[W]), limits = c(3.25, 7.5)) +
        scale_y_continuous(expression(delta*B[e]), limits = c(-2, 2)) +
        theme_bw() +
        theme(text = element_text(size = 15, family = "Cambria"),
              legend.position = c(0.875, 0.5),
              legend.background = element_rect(),
              legend.direction = "vertical",
              legend.key.height = unit(0.75, "cm"))

      ## dS2S vs VS30 ##
      ds2s_VS30_plot <- ggplot(gmpe_res[!duplicated(gmpe_res$StationCode),], aes(x = VS30)) +
        geom_point(aes_string(y = paste("ds2s_", t, sep = "")), color = "black", shape = 1) +
        geom_smooth(aes_string(y = paste("ds2s_", t, sep = "")), color = "red", method = "loess", span = 1) +
        scale_x_log10(expression(V[s30]~(m/s)), breaks = c(180, 360, 760, 1500), limits = c(100, 2000)) +
        scale_y_continuous(expression(delta*S2S[s]), limits = c(-2, 2)) +
        theme_bw() +
        theme(text = element_text(size = 15, family = "Cambria"),
              legend.position = c(0.9, 0.5),
              legend.direction = "vertical",
              legend.key.height = unit(0.75, "cm")) +
        annotation_logticks(sides = "b")

      ## dWSes vs RJB ##
      dwses_RJB_plot <- ggplot(gmpe_res, aes(x = RJB)) +
        geom_point(aes_string(y = paste("dwses_", t, sep = "")), color = "black", shape = 1) +
        stat_summary_bin(aes_string(y = paste("dwses_", t, sep = "")),
                         fun = "mean", bins = 10, color = 'red', size = 1, geom = 'point') +
        stat_summary_bin(aes_string(y = paste("dwses_", t, sep = "")),
                         fun.data = "mean_cl_normal", bins = 10, color = 'red', geom = 'errorbar') +
        scale_x_log10(expression(R[JB]~(km)), limits = c(0.5, 600)) +
        scale_y_continuous(expression(delta*W*S[es]), limits = c(-2, 2)) +
        theme_bw() +
        theme(text = element_text(size = 15, family = "Cambria"),
              legend.title = element_blank(),
              legend.position = "none") +
        annotation_logticks(sides = "b")

      jpeg(file = paste(savefigto, "/res_", t, "_", gmpe, "_", spes, ".jpeg", sep = ""),
           width = 6, height = 9, units = 'in', res = 300)
      multiplot(dbe_M_plot, ds2s_VS30_plot, dwses_RJB_plot)
      dev.off()
    }
  }
  write.csv(gmpe_res, paste(gmpe, "_res_",spes ,domain,alg,".csv", sep=""))
  write.csv(dbe_table0, paste(gmpe, "_dbe_res_",spes,domain,alg,".csv", sep=""))
  write.csv(ds2s_table0, paste(gmpe, "_dS2S_res_",spes,domain,alg,".csv", sep=""))
  
  return(gmpe_res)
}




############ GMPE evaluation ##############
# Generates scenario predictions for a grid of M / distance / depth combinations
# and (optionally) produces diagnostic residual plots and scaling plots.
#
# Arguments:
#   gmpe_res     – flatfile data.frame with at least MAG, RJB, the observed IM columns,
#                  and dbe_/ds2s_ (between-event/site-to-site) residual columns.
#                  Either the augmented output of calculate_residuals() or the
#                  selected_data written by gmm_new() - both are supported, since the
#                  two name these columns differently (by raw period token vs. by IM
#                  string); event_and_site_corrected_<period> (observed IM with dBe and
#                  dS2S removed) is computed on the fly if missing, reconciling either
#                  naming scheme.
#   coeff_table  – coefficient data.frame from gmpe_new().
#   periods      – vector of periods/frequencies.
#   domain       – "SA" or "FAS".
#   spes         – network identifier ("ESM" or other/Japan).
#   do_plot      – if TRUE, writes JPEG scaling plots to savefigto/.
#   savefigto    – output directory for figures (must exist).
#
# Returns: scenarios_allM data.frame with columns (periods, M, D, depth, Pred).
#   NOTE: scenario depth is fixed at 0 km for all predictions; for the ESM model this
#   means h is always taken from the D<10 km bin (h_D10).
gmpe_eval <- function(gmpe_res, coeff_table, periods, domain, spes, gmpe = 'Gmm', savefigto = "GMPE_eval") {
  
  ###### Generate magnitude, distance, response spectra scaling plots #######
  scenarios_allM <- expand.grid("periods" = periods,
                                "M" = c(3.25,3.5,3.75,4,4.25,4.5,
                                        4.6,4.7,4.8,4.9,5,
                                        5.1,5.2,5.3,5.4,5.5,5.6,5.7,5.8,5.9,6,
                                        6.1,6.2,6.3,6.4,6.5,6.6,6.7,6.8,6.9,7,
                                        7.1,7.2,7.3,7.4,7.5),
                                "D" = c(0.5,1,2,3,4,5,7.5,10,12.5,15,20,25,30,35,40,45,50,60,70,80,90,100,
                                        125,150,175,200,250,300,350,400,500,600),
                                "depth" = c(0),
                                "Pred" = c(0))

  if(domain == "FAS") {
    tu <- 'Hz'
  } else {
    tu <- 's'
  }
  
  #### GMPE prediction plots ####

  ## color blind friendly palettes ###
  # The palette with grey:
  cbPalette <- c("#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
  # The palette with black:
  cbbPalette <- c("#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
  
  for(t in periods){
    print(t)

    tt <- ifelse(t == "PGA", 0.01, as.numeric(t))
    IM <- ifelse(t == "PGA", t, sprintf(as.numeric(t), fmt = 'X%#.3f'))

    corr_col <- paste("event_and_site_corrected_", t, sep = "")

    # calculate_residuals() names this column by the raw period token (t) and
    # gmm_new() names it by the IM string instead (and already computes it directly) -
    # reconcile both naming schemes so either function's output can be plotted here.
    if (!(corr_col %in% colnames(gmpe_res))) {
      im_corr_col <- paste("event_and_site_corrected_", IM, sep = "")
      if (im_corr_col %in% colnames(gmpe_res)) {
        gmpe_res[[corr_col]] <- gmpe_res[[im_corr_col]]
      } else {
        dbe_col  <- if (paste("dbe_", t, sep = "") %in% colnames(gmpe_res))  paste("dbe_", t, sep = "")  else paste("dbe_", IM, sep = "")
        ds2s_col <- if (paste("ds2s_", t, sep = "") %in% colnames(gmpe_res)) paste("ds2s_", t, sep = "") else paste("ds2s_", IM, sep = "")
        gmpe_res[[corr_col]] <- exp(log(gmpe_res[[IM]]) - gmpe_res[[dbe_col]] - gmpe_res[[ds2s_col]])
      }
    }

    if (spes=="ESM"){
      scenarios_allM[which(scenarios_allM$periods == t),"Pred"]  <- Kothaetal2020Epe(t,scenarios_allM[which(scenarios_allM$periods == t),"M"],
                                                                                     scenarios_allM[which(scenarios_allM$periods == t),"D"],
                                                                                     scenarios_allM[which(scenarios_allM$periods == t),"depth"],
                                                                                     coeff_table, domain)
    } else {
      scenarios_allM[which(scenarios_allM$periods == t),"Pred"]  <- gmpe_allM_YA15_Mh_Mref(t,scenarios_allM[which(scenarios_allM$periods == t),"M"],
                                                                                 scenarios_allM[which(scenarios_allM$periods == t),"D"],
                                                                                 coeff_table)
    }
    
    ################### RJB and MW with PREDICTION ###############
      ###### Generate magnitude, distance, response spectra scaling plots #######

      ### Distance scaling plot ###
      m_plot <- c(3.5,4.5,5.5,6,6.5,7,7.5)
      
      gmpe_dist_scaling <- ggplot()+
        annotate("text",label = paste("T = ",t,tu,sep = ""), x  = 100,
                 y = 2*max(gmpe_res[[corr_col]], na.rm = T), size = 7.5, family = "Cambria")+
        scale_x_log10(expression(R[JB]~"(km)"), limits = c(0.5,600))+
        scale_y_log10(bquote(.(domain)[(delta*B[e]+delta*S2S[s])]~(g)), labels = comma)+
        scale_colour_gradientn("", colors = cbbPalette,labels = paste("M",m_plot,sep = ""), breaks = m_plot)+
        guides(colour = guide_legend(override.aes = list(shape = 15, size = 6)))+
        theme_bw()+
        theme(text = element_text(size = 20, family = "Cambria"),
              strip.background = element_blank(),
              legend.position = c(0.2,0.2),
              legend.direction = "vertical",
              legend.key.width = unit(1,"cm"))+
        annotation_logticks(sides = "bl")
      
      for(m in m_plot) {
        
        gmpe_dist_scaling <-  gmpe_dist_scaling +
          geom_point(data = subset(gmpe_res, MAG == m+0.1 & MAG >= m-0.1),
                     aes_string(x = "RJB", y = corr_col, color = m),shape = 15)+
          geom_line(data = subset(scenarios_allM, M == m & periods == t), aes(D, Pred, group = M), color = "black", linewidth = 2) +
          geom_line(data = subset(scenarios_allM, M == m & periods == t), aes(D, Pred, group = M, color = M), linewidth = 1)
        
      }
      
      
      ### Magnitude scaling plot ###
      d_plot <- c(5,10,25,50,100,200,400)
      
      gmpe_mag_scaling <- ggplot()+
        annotate("text",label = paste("T = ",t,tu,sep = ""), x  = 4.25,
                 y = 2*max(gmpe_res[[corr_col]], na.rm = T), size = 7.5, family = "Cambria")+
        geom_point(data = gmpe_res, aes_string("MAG", corr_col, color  = "RJB"))+
        scale_x_continuous(expression(M[W]), limits = c(3.25,8.5))+
        scale_y_log10(bquote(.(domain)[(delta*B[e]+delta*S2S[s])]~(g)), labels = comma)+
        scale_colour_viridis(expression(R[JB]), trans = "log", breaks = c(1,5,25,125),
                             guide = guide_colorbar(reverse = TRUE))+
        guides(colour = guide_legend(override.aes = list(shape = 15, size = 6)))+
        theme_bw()+
        theme(text = element_text(size = 20, family = "Cambria"),
              strip.background = element_blank(),
              legend.position = c(0.9,0.6),
              legend.direction = "vertical",
              legend.key.width = unit(0.5,"cm"),
              legend.key.height = unit(1.5,"cm"))+
        annotation_logticks(sides = "l")
      
      for(d in d_plot) {
        
        gmpe_mag_scaling <- gmpe_mag_scaling +
          geom_line(data = subset(scenarios_allM, D == d & periods == t), aes(M, Pred, group = D), color = "black", linewidth = 2) +
          geom_line(data = subset(scenarios_allM, D == d & periods == t), aes(M, Pred, group = D, color = D), linewidth = 1)
        
      }
      jpeg(file = paste(savefigto,"/mag_dist_scaling",t,"_",gmpe,"_",spes,".jpeg",sep=""),
           width = 12, height = 6, units = 'in', res = 300)
      
      multiplot(gmpe_dist_scaling, gmpe_mag_scaling, cols = 2)
      
      dev.off()
    }
  write.csv(scenarios_allM, paste('gmpe_',spes,"_scenario_",domain ,".csv", sep=""))
  return(scenarios_allM)
}

# Multiple plot function
#
# ggplot objects can be passed in ..., or to plotlist (as a list of ggplot objects)
# - cols:   Number of columns in layout
# - layout: A matrix specifying the layout. If present, 'cols' is ignored.
#
# If the layout is something like matrix(c(1,2,3,3), nrow=2, byrow=TRUE),
# then plot 1 will go in the upper left, 2 will go in the upper right, and
# 3 will go all the way across the bottom.
#
multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  library(grid)
  
  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)
  
  numPlots = length(plots)
  
  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                     ncol = cols, nrow = ceiling(numPlots/cols))
  }
  
  if (numPlots==1) {
    print(plots[[1]])
    
  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
    
    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
      
      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}