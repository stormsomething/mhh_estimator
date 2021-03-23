/*
  Copyright (C) 2002-2019 CERN for the benefit of the ATLAS collaboration
*/

// Class handling the output of the MissingMassCalculator
// author Michael Huebner <michael.huebner@no.spam.cern.ch>

// Local include(s):
#include "MissingMassOutput.h"

using namespace DiTauMassTools;

// Default Constructor
MissingMassOutput::MissingMassOutput(){
}

// Default Desctructor
MissingMassOutput::~MissingMassOutput(){
}

//------- clearing output
void MissingMassOutput::ClearOutput(bool fUseVerbose) {

  if(fUseVerbose == 1){ Info("DiTauMassTools", "MissingMassCalculator::ClearOutput()"); }
  FitStatus=0;

  for (int imeth=0; imeth<MMCFitMethodV2::MAX; ++imeth)
    {
      if(fUseVerbose == 1){ Info("DiTauMassTools", "%s", ("MissingMassCalculator::ClearOutput(): clearing for method "+std::to_string(imeth)).c_str()); }
      FitSignificance[imeth] = -1.0;
      FittedMass[imeth] = 0.0;
      nuvec1[imeth].SetPxPyPzE(0.0,0.0,0.0,0.0);
      objvec1[imeth].SetPxPyPzE(0.0,0.0,0.0,0.0);
      nuvec2[imeth].SetPxPyPzE(0.0,0.0,0.0,0.0);
      objvec2[imeth].SetPxPyPzE(0.0,0.0,0.0,0.0);
      totalvec[imeth].SetPxPyPzE(0.0,0.0,0.0,0.0);
      FittedMetVec[imeth].Set(0.0,0.0);
    }

  RMS2MPV = 0.0;

  return;
}


// return fit status
int MissingMassOutput::GetFitStatus() {
  return FitStatus;
}

// returns fit significance
double MissingMassOutput::GetFitSignificance(int fitcode) {
  double signif = -1.0;
  if (fitcode<0 || fitcode >= MMCFitMethodV2::MAX) {
    Error("DiTauMassTools", "%s", ("MissingMassCalculatorV2::GetFitSignificance ERROR ! fitcode="+std::to_string(fitcode)
              +". Should be between 0 and "+std::to_string(MMCFitMethodV2::MAX-1)).c_str());
  } else {
    signif = FitSignificance[fitcode];
  }

  return signif;
}

// returns RMS/MPV according to histogram method
double MissingMassOutput::GetRms2Mpv() {
  return RMS2MPV;
}

// returns fitted Mass
double MissingMassOutput::GetFittedMass(int fitcode) {
  // 0 best parameter space
  // 1 best mass
  // best nu from hist
  double mass = 0.0;
  if (fitcode<0 || fitcode >= MMCFitMethodV2::MAX) {
    Error("DiTauMassTools", "%s", ("MissingMassCalculatorV2::GetFittedMass ERROR ! fitcode="+std::to_string(fitcode)
              +". Should be between 0 and "+std::to_string(MMCFitMethodV2::MAX-1)).c_str());
  } else {
    mass = FittedMass[fitcode];
  }

  return mass;
}

// returns upper error on fitted Mass
double MissingMassOutput::GetFittedMassErrorUp(int fitcode) {
  // 0 best parameter space
  // 1 best mass
  // best nu from hist
  double massUpperError = 0.0;
  if (fitcode<0 || fitcode >= MMCFitMethodV2::MAX) {
    Error("DiTauMassTools", "%s", ("MissingMassCalculatorV2::GetFittedMass ERROR ! fitcode="+std::to_string(fitcode)
              +". Should be between 0 and "+std::to_string(MMCFitMethodV2::MAX-1)).c_str());
  } else {
    massUpperError = FittedMassUpperError[fitcode];
  }

  return massUpperError;
}

// returns upper error on fitted Mass
double MissingMassOutput::GetFittedMassErrorLow(int fitcode) {
  // 0 best parameter space
  // 1 best mass
  // best nu from hist
  double massLowerError = 0.0;
  if (fitcode<0 || fitcode >= MMCFitMethodV2::MAX) {
    Error("DiTauMassTools", "%s", ("MissingMassCalculatorV2::GetFittedMass ERROR ! fitcode="+std::to_string(fitcode)
              +". Should be between 0 and "+std::to_string(MMCFitMethodV2::MAX-1)).c_str());
  } else {
    massLowerError = FittedMassLowerError[fitcode];
  }

  return massLowerError;
}

int MissingMassOutput::GetNTrials()
{
  return NTrials;
}

int MissingMassOutput::GetNSuccesses()
{
  return NSuccesses;
}

int MissingMassOutput::GetNSolutions()
{
  return NSolutions;
}

// sum of weights of all solutions
double MissingMassOutput::GetSumW()
{
  return SumW;
}

// average RMS of solutions in one event
double MissingMassOutput::GetAveSolRMS()
{
  return AveSolRMS;
}

std::shared_ptr<TH1F> MissingMassOutput::GetMassHistogram()
{
  return hMfit_all;
}

std::shared_ptr<TH1F> MissingMassOutput::GetMassHistogramNoWeight()
{
  return hMfit_allNoWeight;
}

// returns neutrino 4-vec
TLorentzVector MissingMassOutput::GetNeutrino4vec(int fitcode, int ind)
{
  TLorentzVector vec(0.0,0.0,0.0,0.0);
  if (fitcode!=MMCFitMethodV2::MAXW && fitcode!=MMCFitMethodV2::MLNU3P )
    {
      Error("DiTauMassTools", "%s", ("MissingMassCalculatorV2::GetNeutrino4Vec ERROR ! fitcode="+std::to_string(fitcode)
                +". Should be either "+std::to_string(MMCFitMethodV2::MAXW)+" or "+std::to_string(MMCFitMethodV2::MLNU3P)).c_str());
    }
  else if (FitStatus>0)
    {
      if(ind==0) vec = nuvec1[fitcode];
      else if(ind==1) vec = nuvec2[fitcode];
    }
  return vec;
}

// returns neutrino 4-vec
TLorentzVector MissingMassOutput::GetTau4vec(int fitcode, int ind)
{
  TLorentzVector vec(0.0,0.0,0.0,0.0);
  if (fitcode!=MMCFitMethodV2::MAXW && fitcode!=MMCFitMethodV2::MLNU3P )
    {
      Error("DiTauMassTools", "%s", ("MissingMassCalculatorV2::GetTau4vec ERROR ! fitcode="+std::to_string(fitcode)
                +". Should be either "+std::to_string(MMCFitMethodV2::MAXW)+" or "+std::to_string(MMCFitMethodV2::MLNU3P)).c_str());
    }
  else if (FitStatus>0)
    {
      if(ind==0) vec = objvec1[fitcode];
      else if(ind==1) vec = objvec2[fitcode];
    }
  return vec;
}

// returns 4-vec for resonance
TLorentzVector MissingMassOutput::GetResonanceVec(int fitcode) {
  TLorentzVector vec(0.0,0.0,0.0,0.0);
  if (fitcode!=MMCFitMethodV2::MAXW && fitcode!=MMCFitMethodV2::MLNU3P )
    {
      Error("DiTauMassTools", "%s", ("MissingMassCalculatorV2::GetResonanceVec ERROR ! fitcode="+std::to_string(fitcode)
                +". Should be either "+std::to_string(MMCFitMethodV2::MAXW)+" or "+std::to_string(MMCFitMethodV2::MLNU3P)).c_str());
    }
  else if (FitStatus>0)
    {
      vec = objvec1[fitcode]+objvec2[fitcode];
    }

  return vec;
}

// returns 2-vec for fitted MET
TVector2 MissingMassOutput::GetFittedMetVec(int fitcode) {
  TVector2 vec(0.0,0.0);
  if (fitcode!=MMCFitMethodV2::MAXW && fitcode!=MMCFitMethodV2::MLNU3P )
    {
      Error("DiTauMassTools", "%s", ("MissingMassCalculatorV2::GetFittedMetVec ERROR ! fitcode="+std::to_string(fitcode)
                +". Should be either "+std::to_string(MMCFitMethodV2::MAXW)+" or "+std::to_string(MMCFitMethodV2::MLNU3P)).c_str());
    }
  else if (FitStatus>0)
    {
      vec=FittedMetVec[fitcode];
    }

  return vec;
}
