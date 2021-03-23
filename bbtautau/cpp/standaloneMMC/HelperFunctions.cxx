/*
  Copyright (C) 2002-2019 CERN for the benefit of the ATLAS collaboration
*/

// vim: ts=2 sw=2

// if included, do not use fast calculation of sin/cos but built-in functions
//#define NOFASTSINCOS

// local include(s)
#include "HelperFunctions.h"

using namespace DiTauMassTools;

int DiTauMassTools::getFirstBinAboveMax(const std::shared_ptr<TH1F>& hist, double max, double targetVal){
  int maxBin = hist->FindBin(max);
  int targetBin = maxBin;
  for(int i=maxBin; i<=hist->GetNbinsX(); i++){
    if(hist->GetBinContent(i)<targetVal){
      targetBin = i-1;
      break;
    }
  }
  return targetBin;
}

int DiTauMassTools::getFirstBinBelowMax(const std::shared_ptr<TH1F>& hist, double max, double targetVal){
  int maxBin = hist->FindBin(max);
  int targetBin = maxBin;
  for(int i=1; i<=maxBin; i++){
    if(hist->GetBinContent(i)>=targetVal){
      targetBin = i;
      break;
    }
  }
  return targetBin;
}

double DiTauMassTools::MaxDelPhi(int tau_type, double Pvis, double dRmax_tau)
{
  return dRmax_tau+0*Pvis*tau_type; // hack to avoid warning
}

double DiTauMassTools::Angle(const TLorentzVector & vec1, const TLorentzVector & vec2) {
  //SpeedUp (both are equivalent in fact)
  return acos((vec1.Px()*vec2.Px()+vec1.Py()*vec2.Py()+vec1.Pz()*vec2.Pz())/(vec1.P()*vec2.P()));
  //return vec1.Angle(vec2.Vect());
}

//put back phi within -pi, +pi
double DiTauMassTools::fixPhiRange  (const double & phi)
{
  double phiOut=phi;

  if (phiOut>0){
    while (phiOut>TMath::Pi()) {
      phiOut-=TMath::TwoPi();
    }
  }

  else{
    while (phiOut<-TMath::Pi()) {
      phiOut+=TMath::TwoPi();
    }
  }
  return phiOut;
}

// fast approximate calculation of sin and cos
// approximation good to 1 per mill. s²+c²=1 strictly exact though
// it is like using slightly different values of phi1 and phi2
void DiTauMassTools::fastSinCos (const double & phiInput, double & sinPhi, double & cosPhi)
{
  const double fastB=4/TMath::Pi();
  const double fastC=-4/(TMath::Pi()*TMath::Pi());
  const double fastP=9./40.;
  const double fastQ=31./40.;
  // use normal sin cos if switch off
#ifdef NOFASTSINCOS
  sinPhi=sin(phiInput);
  cosPhi=cos(phiInput);
#else


  double phi = fixPhiRange(phiInput);

  // http://devmaster.net/forums/topic/4648-fast-and-accurate-sinecosine/
  // accurate to 1 per mille

  // even faster
  //  const double y=fastB*phi+fastC*phi*std::abs(phi);
  // sinPhi=fastP*(y*std::abs(y)-y)+y;
  const double y=phi*(fastB+fastC*std::abs(phi));
  sinPhi=y*(fastP*std::abs(y)+fastQ);


  //note that one could use cos(phi)=sin(phi+pi/2), however then one would not have c²+s²=1 (would get it only within 1 per mille)
  // the choice here is to keep c^2+s^2=1 so everything is as one would compute c and s from a slightly (1 per mille) different angle
  cosPhi=sqrt(1-std::pow(sinPhi,2));
  if (std::abs(phi)>TMath::PiOver2()) cosPhi=-cosPhi;
#endif
}

// return true if cache was uptodate
bool DiTauMassTools::updateDouble  (const double in, double & out)
{
  if (out==in) return true;
  out=in;
  return false;
}

//CP::CorrectionCode MissingMassToolV2::getLFVMode( const xAOD::IParticle* p1, const xAOD::IParticle* p2, int mmcType1, int mmcType2) {
int DiTauMassTools::getLFVMode( const int part1, const int part2, int mmcType1, int mmcType2) {
  return part1;
}

int DiTauMassTools::mmcType(const int part)
{
  return part;
}
//________________________________________________________________________

double DiTauMassTools::mT(const TLorentzVector & vec,const TVector2 & met_vec) {
  double mt=0.0;
  double dphi=fabs(TVector2::Phi_mpi_pi(vec.Phi()-met_vec.Phi()));
  double cphi=1.0-cos(dphi);
  if(cphi>0.0) mt=sqrt(2.0*vec.Pt()*met_vec.Mod()*cphi);
  return mt;
}

//________________________________________________________________________
void DiTauMassTools::readInParams(TDirectory* dir, MMCCalibrationSetV2::e aset, std::vector<TF1*>& lep_numass, std::vector<TF1*>& lep_angle, std::vector<TF1*>& lep_ratio, std::vector<TF1*>& had_angle, std::vector<TF1*>& had_ratio) {
	std::string paramcode;
	if (aset == MMCCalibrationSetV2::MMC2019) paramcode = "MMC2019MC16";
	else {
		Info("DiTauMassTools", "The specified calibration version does not support root file parametrisations");
		return;
	}
	TIter next(dir->GetListOfKeys());
	TKey* key;
	while ((key = (TKey*)next())) {
		TClass *cl = gROOT->GetClass(key->GetClassName());
		// if there is another subdirectory, go into that dir
		if (cl->InheritsFrom("TDirectory")) {
			dir->cd(key->GetName());
			TDirectory *subdir = gDirectory;
			readInParams(subdir, aset, lep_numass, lep_angle, lep_ratio, had_angle, had_ratio);
			dir->cd();
		}
		else if (cl->InheritsFrom("TF1") || cl->InheritsFrom("TGraph")) {
			// get parametrisations and sort them into their corresponding vectors
			std::string total_path = dir->GetPath();
			if (total_path.find(paramcode) == std::string::npos) continue;
			TF1* func = (TF1*)dir->Get( (const char*) key->GetName() );
			TF1* f = new TF1(*func);
			if (total_path.find("lep") != std::string::npos)
			{
				if (total_path.find("Angle") != std::string::npos){
					lep_angle.push_back(f);
				}
				else if (total_path.find("Ratio") != std::string::npos)
				{
					lep_ratio.push_back(f);
				}
				else if (total_path.find("Mass") != std::string::npos)
				{
					lep_numass.push_back(f);
				}
				else
				{
					Warning("DiTauMassTools", "Undefined leptonic PDF term in input file.");
				}
			}
			else if (total_path.find("had") != std::string::npos)
			{
				if (total_path.find("Angle") != std::string::npos){
					had_angle.push_back(f);
				}
				else if (total_path.find("Ratio") != std::string::npos)
				{
					had_ratio.push_back(f);
				}
				else
				{
					Warning("DiTauMassTools", "Undefined hadronic PDF term in input file.");
				}
			}
			else
			{
				Warning("DiTauMassTools", "Undefined decay channel in input file.");
			}
		}
		else {
			Warning("DiTauMassTools", "Class in input file not recognized.");
		}
	}
}
