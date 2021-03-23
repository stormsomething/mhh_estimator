/*
  Copyright (C) 2002-2019 CERN for the benefit of the ATLAS collaboration
*/

// Asg wrapper around the MissingMassCalculator
// author Michael Huebner <michael.huebner@no.spam.cern.ch>
#ifndef DITAUMASSTOOLS_HELPERFUNCTIONS_H
#define DITAUMASSTOOLS_HELPERFUNCTIONS_H

// local include(s)
#include "TH1F.h"
#include "TF1.h"
#include "TKey.h"
#include "TDirectory.h"
#include "TROOT.h"

namespace DiTauMassTools{

namespace MaxHistStrategyV2
{
  enum e { MAXBIN=0,MAXBINWINDOW, SLIDINGWINDOW, FIT,MAXMAXHISTSTRATEGY };
}

namespace HistInfoV2
{
  enum e { PROB=0, INTEGRAL, CHI2, DISCRI, TANTHETA, TANTHETAW, FITLENGTH, RMS, RMSVSDISCRI, MEANBIN, MAXHISTINFO };
}

namespace MMCCalibrationSetV2
{
  enum e { MMC2011=0, MMC2012, MMC2015, MMC2015HIGHMASS, UPGRADE, LFVMMC2012, MMC2016MC15C, MMC2019, MAXMMCCALIBRATIONSET };
  const std::string name[MAXMMCCALIBRATIONSET]={ "MMC2011", "MMC2012", "MMC2015", "MMC2015HIGHMASS", "UPGRADE", "LFVMMC2012", "MMC2016MC15C", "MMC2019"};
}

namespace MMCFitMethodV2
{
  enum e { MAXW=0, MLM, MLNU3P,MAX};
  const std::string name[MAX]={ "MAXW=MaximumWeight", "MLM=MostLikelyMass", "MLNU3P=MostLikelyNeUtrino3Momentum"};
  const std::string shortName[MAX]={ "MAXW", "MLM", "MLNU3P"};
}

namespace TauTypes
{
  enum e {ll=0, lh, hh};
}

// define ignore template to suppress warnings in MissingMassProb
// see source file of MissingMassProb for further reasoning
template <typename T> void ignore(T &&){}

int getFirstBinBelowMax(const std::shared_ptr<TH1F>& hist, double max, double targetVal);
int getFirstBinAboveMax(const std::shared_ptr<TH1F>& hist, double max, double targetVal);
double Angle(const TLorentzVector & vec1, const TLorentzVector & vec2);
bool updateDouble  (const double in, double & out) ;
void fastSinCos (const double & phi, double & sinPhi, double & cosPhi);
double fixPhiRange (const double & phi);
double MaxDelPhi(int tau_type, double Pvis, double dRmax_tau);
int getLFVMode( const int part1, const int part2, int mmcType1, int mmcType2);
int mmcType(const int part); // returns particle type as required by MMC
double mT(const TLorentzVector & vec,const TVector2 & met_vec);
void readInParams(TDirectory* dir, MMCCalibrationSetV2::e aset, std::vector<TF1*>& lep_numass, std::vector<TF1*>& lep_angle, std::vector<TF1*>& lep_ratio, std::vector<TF1*>& had_angle, std::vector<TF1*>& had_ratio);
} // namespace DiTauMassTools

#endif // not DITAUMASSTOOLS_HELPERFUNCTIONS_H
