/*
This macro shows how to compute jet energy scale.
root -l examples/Example4.C'("delphes_output.root", "plots.root")'
*/

#ifdef __CLING__
R__LOAD_LIBRARY(libDelphes)
#include "classes/DelphesClasses.h"
#include "external/ExRootAnalysis/ExRootTreeReader.h"
#include "external/ExRootAnalysis/ExRootResult.h"
#else
class ExRootTreeReader;
class ExRootResult;
#endif

class ExRootResult;
class ExRootTreeReader;

//------------------------------------------------------------------------------

void AnalyseEvents(ExRootTreeReader *treeReader,  const char *outputFile_part)
{
  TClonesArray *branchGenJet = treeReader->UseBranch("GenJet");
  TClonesArray *branchParticle = treeReader->UseBranch("Particle");
  TClonesArray *branchEvent = treeReader->UseBranch("Event");
  TClonesArray *branchJet = treeReader->UseBranch("Jet");
  TClonesArray *branchEFlowTrack = treeReader->UseBranch("EFlowTrack");
  TClonesArray *branchEFlowPhoton = treeReader->UseBranch("EFlowPhoton");
  TClonesArray *branchEFlowNeutralHadron = treeReader->UseBranch("EFlowNeutralHadron");
  
  Long64_t allEntries = treeReader->GetEntries();
  ofstream myfile_det;
  ofstream myfile_part;

  cout << "** Chain contains " << allEntries << " events" << endl;

  Jet  *genjet;
  Jet *jet;
  GenParticle *particle;
  GenParticle *muparticle;
  GenParticle *genparticle;
  GenParticle *motherparticle;
  TObject *object;
  Photon *photon;
  Track *track;
  Tower *tower;
  
  TLorentzVector genJetMomentum;
  TLorentzVector jetMomentum;
  TLorentzVector myMomentum;
  
  TLorentzVector incomingelectron;
  TLorentzVector incomingpositron;
  TLorentzVector outgoingphoton;
  
  Long64_t entry;

  Int_t i, j;

  myfile_part.open (outputFile_part);

  // Loop over all events
  for(entry = 0; entry < allEntries; ++entry)
  {
    //if (entry > 10000) break;
    // Load selected branches with data from specified event
    treeReader->ReadEntry(entry);
    HepMCEvent *event = (HepMCEvent*) branchEvent -> At(0);
    //std::cout << "weight : " << event->Weight << std::endl;
    
    if(entry%500 == 0) cout << "Event number: "<< entry <<endl;

    //myfile_part << event->Weight << " ";

    for(j = 0; j < branchParticle->GetEntriesFast(); ++j){
        genparticle = (GenParticle*) branchParticle->At(j);
        if (genparticle->PID!=23) continue;
	if (genparticle->Status < 60) continue;
        //std::cout << " " << j << " " << genparticle->PID << " " << genparticle->Status << " " << genparticle->P4().Pt() << std::endl;              
	if (genparticle->P4().Pt() > 200){
	  if (branchJet->GetEntriesFast() > 0){
	    jet = (Jet*) branchJet->At(0);
	    myfile_part << entry << " reco " << genparticle->P4().Pt() << " " << jet->PT << " " << jet->P4().Rapidity() << " " << jet->Phi << " " << jet->P4().M() << " " << jet->Tau[0] << " " << jet->Tau[1] << " " << jet->Tau[2] << " " << jet->Tau[3] << " " << jet->Tau[4] << " " << jet->SoftDroppedP4->M() << " " << jet->SoftDroppedSubJet2.Pt() / (jet->SoftDroppedSubJet1.Pt() + jet->SoftDroppedSubJet2.Pt()) << " " << jet->Constituents.GetEntriesFast() << std::endl;
	  }
	  if (branchGenJet->GetEntriesFast() > 0){
	    jet = (Jet*) branchGenJet->At(0);
	    myfile_part << entry << " truth " << genparticle->P4().Pt() << " " << jet->PT << " " << jet->P4().Rapidity() << " " << jet->Phi << " " << jet->P4().M() << " " << jet->Tau[0]<< " " << jet->Tau[1] << " " << jet->Tau[2] << " " << jet->Tau[3] << " " << jet->Tau[4] << " " << jet->SoftDroppedP4->M() << " " << jet->SoftDroppedSubJet2.Pt() / (jet->SoftDroppedSubJet1.Pt() + jet->SoftDroppedSubJet2.Pt()) << " " << jet->Constituents.GetEntriesFast() << std::endl;
	  }
	}
    }
    //myfile_part << std::endl;
  }
  
}
//------------------------------------------------------------------------------

void myprocess_Omni(const char *inputFile, const char *outputFile_part)
{
  gSystem->Load("libDelphes");

  TChain *chain = new TChain("Delphes");
  chain->Add(inputFile);

  ExRootTreeReader *treeReader = new ExRootTreeReader(chain);
  ExRootResult *result = new ExRootResult();

  AnalyseEvents(treeReader,outputFile_part);

  cout << "** Exiting..." << endl;

  delete result;
  delete treeReader;
  delete chain;
}

//------------------------------------------------------------------------------
