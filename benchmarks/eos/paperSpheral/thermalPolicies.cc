// Standalone thermal-rate research overlay. Native source and old DSO unchanged.
// Opt-in 2D paper-formula overlay. Native Spheral binaries remain untouched.
// Frontiere et al. arXiv:1605.00725v2: Eqs 63,75,76. No density/energy floors.
#include "Geometry/Dimension.hh"
#include "CRKSPH/CRKSPHBase.hh"
#include "Hydro/SpecificThermalEnergyPolicy.hh"
#include "Hydro/HydroFieldNames.hh"
#include "Hydro/entropyWeightingFunction.hh"
#include "RK/RKCorrections.hh"
#include "RK/ReproducingKernel.hh"
#include "RK/RKFieldNames.hh"
#include "RK/computeRKSumVolume.hh"
#include "DataBase/DataBase.hh"
#include "DataBase/State.hh"
#include "DataBase/StateDerivatives.hh"
#include "DataBase/IncrementState.hh"
#include "Neighbor/PairwiseField.hh"
#include "NodeList/FluidNodeList.hh"
#include "Material/EquationOfState.hh"
#include "Boundary/Boundary.hh"
#include "ArtificialViscosity/LimitedMonaghanGingoldViscosityView.hh"
#include "ArtificialViscosity/ArtificialViscosity.hh"
#include <atomic>
#include <vector>
#include <cmath>
#include <limits>

namespace {
std::atomic<unsigned long long> counts[5]{}; // volume,density,final energy,raw predictor,RHS cache
std::vector<std::vector<double>> stageEntropy;
std::vector<std::vector<double>> stageVelocity;
std::vector<double> firstDensityAudit;
std::vector<double> firstRHSAudit;
std::vector<double> avGradientAudit;
std::vector<double> avAffineAudit;
std::vector<double> heatingNodeAudit, heatingPairAudit, partitionAudit;
bool averagedHeating=false;
bool entropyThermal=false; // Research-only continuous Eq63 work distribution.
double thermalFraction(double si,double sj,double power) {
  const double sum=std::abs(si)+std::abs(sj);
  return sum>0. ? (power>=0. ? std::abs(sj) : std::abs(si))/sum : .5;
}
int avGradientMode=0; // 0=native lagged CRK, 1=native fresh SPH, 2=fresh CRK
bool freshRK2=false;
bool dudtPartition=false;
unsigned energyCallsThisStep=0, rhsCallsThisStep=0;
}
extern "C" double warpsph_thermal_fraction(double si,double sj,double power) {return thermalFraction(si,sj,power);}
extern "C" void warpsph_paper_entropy_thermal(int enabled) {entropyThermal=bool(enabled);}
extern "C" void warpsph_paper_fresh_rk2(int enabled) {freshRK2=bool(enabled);}
extern "C" void warpsph_paper_dudt_partition(int enabled) {dudtPartition=bool(enabled);}
extern "C" void warpsph_paper_av_gradient(int mode) {avGradientMode=mode;}
extern "C" void warpsph_paper_averaged_heating(int enabled) {averagedHeating=bool(enabled);}
extern "C" unsigned long long warpsph_paper_heating_node_rows() {return heatingNodeAudit.size()/40;}
extern "C" const double* warpsph_paper_heating_node_data() {return heatingNodeAudit.data();}
extern "C" unsigned long long warpsph_paper_heating_pair_rows() {return heatingPairAudit.size()/16;}
extern "C" const double* warpsph_paper_heating_pair_data() {return heatingPairAudit.data();}
extern "C" unsigned long long warpsph_paper_partition_rows() {return partitionAudit.size()/17;}
extern "C" int warpsph_paper_partition_columns() {return 17;}
extern "C" const double* warpsph_paper_partition_data() {return partitionAudit.data();}
extern "C" unsigned long long warpsph_paper_av_gradient_rows() {return avGradientAudit.size()/13;}
extern "C" const double* warpsph_paper_av_gradient_data() {return avGradientAudit.data();}
extern "C" unsigned long long warpsph_paper_av_affine_rows() {return avAffineAudit.size()/23;}
extern "C" const double* warpsph_paper_av_affine_data() {return avAffineAudit.data();}
extern "C" unsigned long long warpsph_paper_policy_calls(int i) {return counts[i].load();}
extern "C" int warpsph_paper_policy_version() {return 1;}
extern "C" unsigned long long warpsph_paper_audit_rows() {return firstDensityAudit.size()/15;}
extern "C" const double* warpsph_paper_audit_data() {return firstDensityAudit.data();}
extern "C" unsigned long long warpsph_paper_rhs_audit_rows() {return firstRHSAudit.size()/15;}
extern "C" const double* warpsph_paper_rhs_audit_data() {return firstRHSAudit.data();}
extern "C" double warpsph_paper_entropy_fraction(double si,double sj,double work) {
  return Spheral::entropyWeighting(si,sj,work);
}

namespace Spheral {
using D=Dim<2>;
using Vec=D::Vector;
using Sym=D::SymTensor;
using Tensor=D::Tensor;
template<> void SpecificThermalEnergyPolicy<D>::updateAsIncrement(const std::string&,State<D>&,
    StateDerivatives<D>&,double,double,double);

// The same scatter gradient accumulated by CRKSPH::evaluateDerivatives (Eq54),
// evaluated separately before Q so its inputs are at the current RHS state.
void freshCRKGradient(const DataBase<D>& db,const State<D>& state,
    const FieldList<D,Vec>& velocity,FieldList<D,Tensor>& gradient) {
  const auto x=state.fields(HydroFieldNames::position,Vec::zero());
  const auto H=state.fields(HydroFieldNames::H,Sym::zero());
  const auto vol=state.fields(HydroFieldNames::volume,0.);
  const auto c=state.fields(RKFieldNames::rkCorrections(RKOrder::LinearOrder),RKCoefficients<D>());
  const auto& wr=state.get<ReproducingKernel<D>>(RKFieldNames::reproducingKernel(RKOrder::LinearOrder));
  gradient.Zero();
  for(const auto& p:db.connectivityMap().nodePairList()) {
    const auto a=p.i_list,b=p.j_list,i=p.i_node,j=p.j_node;
    const auto r=x(a,i)-x(b,j),dv=velocity(a,i)-velocity(b,j);
    const auto gj=std::get<1>(wr.evaluateKernelAndGradient(r,H(b,j),c(a,i)));
    const auto gi=std::get<1>(wr.evaluateKernelAndGradient(-r,H(a,i),c(b,j)));
    gradient(a,i)-=vol(b,j)*dv.dyad(gj);
    gradient(b,j)+=vol(a,i)*dv.dyad(gi);
  }
}

// Static manufactured fields on the actual initialized geometry. No state is
// changed, and Q is the compiled native Spheral pair routine, not a reimplementation.
void auditAffineAV(const DataBase<D>& db,const State<D>& state) {
  if(not avAffineAudit.empty()) return;
  const auto x=state.fields(HydroFieldNames::position,Vec::zero());
  const auto H=state.fields(HydroFieldNames::H,Sym::zero());
  const auto rho=state.fields(HydroFieldNames::massDensity,0.);
  const auto cs=state.fields(HydroFieldNames::soundSpeed,0.);
  auto vel=db.newFluidFieldList(Vec::zero(),"affine audit v");
  auto grad=db.newFluidFieldList(Tensor::zero(),"affine audit CRK gradient");
  auto exact=db.newFluidFieldList(Tensor::zero(),"affine audit exact gradient");
  auto zero=db.newFluidFieldList(Tensor::zero(),"affine audit zero gradient");
  FieldList<D,double> multipliers;
  LimitedMonaghanGingoldViscosityView<D> q(2.,1.,false,false,1.,.2);
  q.epsilon2(.01);
  const auto& pairs=db.connectivityMap().nodePairList();
  const auto stride=std::max(size_t(1),pairs.size()/1024);
  for(int pattern=0;pattern<2;++pattern) {
    const Tensor G=pattern==0?Tensor(-1.,0.,0.,-.5):Tensor(0.,1.,0.,0.);
    for(auto k=0u;k<vel.size();++k)for(auto i=0u;i<vel[k]->numElements();++i) {
      vel(k,i)=G*x(k,i);exact(k,i)=G;
    }
    freshCRKGradient(db,state,vel,grad);
    for(size_t n=0;n<pairs.size();n+=stride) {
      const auto& p=pairs[n];const auto a=p.i_list,b=p.j_list,i=p.i_node,j=p.j_node;
      if(i>=x[a]->numInternalElements() or j>=x[b]->numInternalElements())continue;
      const auto r=x(a,i)-x(b,j);const auto ei=H(a,i)*r,ej=H(b,j)*r;
      const auto nPerh=.5*(x[a]->nodeList().nodesPerSmoothingScale()+x[b]->nodeList().nodesPerSmoothingScale());
      if(std::min(ei.magnitude(),ej.magnitude())<1./nPerh)continue; // exclude close-pair fold, Eq51
      double qi[3],qj[3],pi,pj;
      auto evaluate=[&](FieldList<D,Tensor>& g,int mode) {
        q.QPiij(pi,pj,qi[mode],qj[mode],a,i,b,j,x(a,i),H(a,i),ei,vel(a,i),rho(a,i),cs(a,i),
            x(b,j),H(b,j),ej,vel(b,j),rho(b,j),cs(b,j),multipliers.view(),multipliers.view(),g.view());
      };
      evaluate(zero,0);evaluate(exact,1);evaluate(grad,2);
      const auto dv=vel(a,i)-vel(b,j);
      const double row[]={double(pattern),double(a),double(b),std::min(ei.magnitude(),ej.magnitude()),nPerh,
          2./H(a,i).Trace(),2./H(b,j).Trace(),r.x(),r.y(),dv.x(),dv.y(),rho(a,i),rho(b,j),cs(a,i),cs(b,j),
          qi[0],qj[0],qi[1],qj[1],qi[2],qj[2],(grad(a,i)-G).maxAbsElement(),(grad(b,j)-G).maxAbsElement()};
      avAffineAudit.insert(avAffineAudit.end(),std::begin(row),std::end(row));
    }
  }
}

template<> void computeRKSumVolume<D>(const ConnectivityMap<D>& cm,const TableKernel<D>& W,
    const FieldList<D,Vec>& x,const FieldList<D,double>& mass,
    const FieldList<D,Sym>& H,FieldList<D,double>& volume) {
  ++counts[0];volume=0.;
  const auto& pairs=cm.nodePairList();
  // Pair reduction is serial here: deterministic qualification implementation.
  // Spheral force, RK and energy loops retain OpenMP.
  for (const auto& p:pairs) {
    const auto r=x(p.i_list,p.i_node)-x(p.j_list,p.j_node);
    const auto& Hi=H(p.i_list,p.i_node);const auto& Hj=H(p.j_list,p.j_node);
    volume(p.i_list,p.i_node)+=W.kernelValue((Hi*r).magnitude(),Hi.Determinant());
    volume(p.j_list,p.j_node)+=W.kernelValue((Hj*r).magnitude(),Hj.Determinant());
  }
  for (auto k=0u;k<volume.size();++k) for(auto i=0u;i<volume[k]->numInternalElements();++i) {
    volume(k,i)=1./(volume(k,i)+W.kernelValue(0.,H(k,i).Determinant()));
  }
}

template<> void CRKSPHBase<D>::preStepInitialize(const DataBase<D>& db,State<D>& state,StateDerivatives<D>& derivs) {
  ++counts[1];
  energyCallsThisStep=rhsCallsThisStep=0;
  VERIFY2(densityUpdate()==MassDensityType::RigorousSumDensity,"Paper overlay expects RigorousSumDensity dispatch");
  const auto& wr=state.get<ReproducingKernel<D>>(RKFieldNames::reproducingKernel(correctionOrder()));
  const auto& cm=db.connectivityMap();
  auto rho=state.fields(HydroFieldNames::massDensity,0.);
  const auto mass=state.fields(HydroFieldNames::mass,0.);
  const auto vol=state.fields(HydroFieldNames::volume,0.);
  const auto x=state.fields(HydroFieldNames::position,Vec::zero());
  const auto H=state.fields(HydroFieldNames::H,Sym::zero());
  auto c0=state.fields(RKFieldNames::rkCorrections(RKOrder::ZerothOrder),RKCoefficients<D>());
  auto c=state.fields(RKFieldNames::rkCorrections(correctionOrder()),RKCoefficients<D>());
  // Volumes have just been recomputed. Density must use corrections for those volumes.
  wr.computeCorrections(cm,vol,x,H,false,c0,c);
  for(auto* bc:this->boundaryConditions()) {bc->applyFieldListGhostBoundary(c0);bc->applyFieldListGhostBoundary(c);}
  for(auto* bc:this->boundaryConditions()) bc->finalizeGhostBoundary();
  auto num=db.newFluidFieldList(0.,"paper density numerator");
  auto den=db.newFluidFieldList(0.,"paper density denominator");
  for(const auto& p:cm.nodePairList()) {
    const auto a=p.i_list,b=p.j_list,i=p.i_node,j=p.j_node;
    const auto r=x(a,i)-x(b,j);
    const double wij=wr.evaluateKernel(r,H(b,j),c(a,i));
    const double wji=wr.evaluateKernel(-r,H(a,i),c(b,j));
    num(a,i)+=(a==b?mass(b,j):mass(a,i))*vol(b,j)*wij;
    den(a,i)+=vol(b,j)*vol(b,j)*wij;
    num(b,j)+=(a==b?mass(a,i):mass(b,j))*vol(a,i)*wji;
    den(b,j)+=vol(a,i)*vol(a,i)*wji;
  }
  for(auto k=0u;k<rho.size();++k) for(auto i=0u;i<rho[k]->numInternalElements();++i) {
    const double self=wr.evaluateKernel(Vec::zero(),H(k,i),c(k,i));
    num(k,i)+=mass(k,i)*vol(k,i)*self;den(k,i)+=vol(k,i)*vol(k,i)*self;
    rho(k,i)=num(k,i)/den(k,i); // Eq76; intentionally no native rho/H clipping.
  }
  for(auto* bc:this->boundaryConditions()) bc->applyFieldListGhostBoundary(rho);
  for(auto* bc:this->boundaryConditions()) bc->finalizeGhostBoundary();
  // Refresh EOS after the prescribed density replacement, before selecting dt/RHS.
  auto pressure=state.fields(HydroFieldNames::pressure,0.);
  auto cs=state.fields(HydroFieldNames::soundSpeed,0.);
  auto entropy=state.fields(HydroFieldNames::entropy,0.);
  const auto eps=state.fields(HydroFieldNames::specificThermalEnergy,0.);
  for(auto k=0u;k<rho.size();++k) {
    const auto& n=dynamic_cast<const FluidNodeList<D>&>(rho[k]->nodeList());
    n.pressure(*pressure[k]);n.soundSpeed(*cs[k]);n.equationOfState().setEntropy(*entropy[k],*rho[k],*eps[k]);
  }
  if(counts[1].load()==1) {
    for(auto k=0u;k<rho.size();++k)for(auto i=0u;i<rho[k]->numElements();++i) {
      const double row[]={x(k,i).x(),x(k,i).y(),mass(k,i),vol(k,i),H(k,i).xx(),H(k,i).xy(),H(k,i).yy(),rho(k,i),
          c(k,i)[0],c(k,i)[1],c(k,i)[2],double(k),double(i<rho[k]->numInternalElements()),pressure(k,i),eps(k,i)};
      firstDensityAudit.insert(firstDensityAudit.end(),std::begin(row),std::end(row));
    }
  }
}

template<> bool RKCorrections<D>::initialize(const double time,const double dt,const DataBase<D>& db,
    State<D>& state,StateDerivatives<D>& derivs) {
  // Eq75 must also hold on the interim RHS. Native Spheral otherwise evolves
  // this field with its bounded continuity-volume policy between step resets.
  auto vol=state.fields(HydroFieldNames::volume,0.);
  const auto x=state.fields(HydroFieldNames::position,Vec::zero());
  const auto H=state.fields(HydroFieldNames::H,Sym::zero());
  const auto mass=state.fields(HydroFieldNames::mass,0.);
  computeRKSumVolume<D>(db.connectivityMap(),mWR.at(RKOrder::LinearOrder).kernel(),x,mass,H,vol);
  for(auto* bc:this->boundaryConditions())bc->applyFieldListGhostBoundary(vol);
  for(auto* bc:this->boundaryConditions())bc->finalizeGhostBoundary();
  if(mUpdateInStep) updateCorrections(db,state);
  if(avGradientMode==2) {
    const auto velocity=state.fields(HydroFieldNames::velocity,Vec::zero());
    auto gradient=state.fields(HydroFieldNames::ArtificialViscosityVelocityGradient,Tensor::zero());
    freshCRKGradient(db,state,velocity,gradient);
    for(auto* bc:this->boundaryConditions())bc->applyFieldListGhostBoundary(gradient);
    for(auto* bc:this->boundaryConditions())bc->finalizeGhostBoundary();
  }
  if(counts[1].load()==1) auditAffineAV(db,state);
  const auto s=state.fields(HydroFieldNames::entropy,0.);
  stageEntropy.resize(s.size());
  stageVelocity.resize(s.size());
  const auto velocity=state.fields(HydroFieldNames::velocity,Vec::zero());
  for(auto k=0u;k<s.size();++k) {
    stageEntropy[k].resize(s[k]->numElements());
    stageVelocity[k].resize(2*s[k]->numElements());
    for(auto i=0u;i<s[k]->numElements();++i) {
      stageEntropy[k][i]=s(k,i);
      stageVelocity[k][2*i]=velocity(k,i).x();stageVelocity[k][2*i+1]=velocity(k,i).y();
    }
  }
  ++counts[4];
  ++rhsCallsThisStep;
  // The first CheapRK2 prediction starts with zero derivative fields. Capture
  // the second midpoint, where particle positions/supports actually move.
  if(firstRHSAudit.empty() and counts[1].load()>=2 and rhsCallsThisStep==2) {
    const auto rho=state.fields(HydroFieldNames::massDensity,0.);
    const auto c=state.fields(RKFieldNames::rkCorrections(RKOrder::LinearOrder),RKCoefficients<D>());
    const auto pressure=state.fields(HydroFieldNames::pressure,0.);
    const auto eps=state.fields(HydroFieldNames::specificThermalEnergy,0.);
    for(auto k=0u;k<rho.size();++k)for(auto i=0u;i<rho[k]->numElements();++i) {
      const double row[]={x(k,i).x(),x(k,i).y(),mass(k,i),vol(k,i),H(k,i).xx(),H(k,i).xy(),H(k,i).yy(),rho(k,i),
          c(k,i)[0],c(k,i)[1],c(k,i)[2],double(k),double(i<rho[k]->numInternalElements()),pressure(k,i),eps(k,i)};
      firstRHSAudit.insert(firstRHSAudit.end(),std::begin(row),std::end(row));
    }
  }
  return true;
}

// Observe the completed RHS before SynchronousRK2 restores state0 for its final
// update. Looking at the AV state field from energy::update is too late there.
// Preserve the native compatible-energy boundary finalization verbatim.
template<> void CRKSPHBase<D>::finalizeDerivatives(const double,const double,
    const DataBase<D>& db,const State<D>& state,StateDerivatives<D>& derivs) const {
  const bool audit=counts[1].load()==2;
  if(averagedHeating or entropyThermal or audit) {
    // Native Eq78 uses Q_j in du_i. This opt-in experiment changes only that
    // local share to (Q_i+Q_j)/2, preserving the instantaneous pair sum.
    const auto x=state.fields(HydroFieldNames::position,Vec::zero());
    const auto H=state.fields(HydroFieldNames::H,Sym::zero());
    const auto v=state.fields(HydroFieldNames::velocity,Vec::zero());
    const auto m=state.fields(HydroFieldNames::mass,0.);
    const auto rho=state.fields(HydroFieldNames::massDensity,0.);
    const auto P=state.fields(HydroFieldNames::pressure,0.);
    const auto u=state.fields(HydroFieldNames::specificThermalEnergy,0.);
    const auto cs=state.fields(HydroFieldNames::soundSpeed,0.);
    const auto V=state.fields(HydroFieldNames::volume,0.);
    const auto c=state.fields(RKFieldNames::rkCorrections(RKOrder::LinearOrder),RKCoefficients<D>());
    auto g=state.fields(HydroFieldNames::ArtificialViscosityVelocityGradient,Tensor::zero());
    const auto acc=derivs.fields(HydroFieldNames::hydroAcceleration,Vec::zero());
    auto du=derivs.fields(IncrementState<D,double>::prefix()+HydroFieldNames::specificThermalEnergy,0.);
    auto native=db.newFluidFieldList(0.,"heating audit native");
    auto average=db.newFluidFieldList(0.,"heating audit average");
    auto delta=db.newFluidFieldList(0.,"heating policy delta");
    auto thermal=db.newFluidFieldList(0.,"instantaneous entropy thermal rate");
    const auto entropy=state.fields(HydroFieldNames::entropy,0.);
    const auto& W=state.get<ReproducingKernel<D>>(RKFieldNames::reproducingKernel(RKOrder::LinearOrder));
    auto& handle=this->artificialViscosity();
    VERIFY2(handle.QPiTypeIndex()==std::type_index(typeid(double)),"Heating control requires scalar Q");
    auto q=handle.getScalarView();
    auto fCl=state.fields(HydroFieldNames::ArtificialViscousClMultiplier,0.,true);
    auto fCq=state.fields(HydroFieldNames::ArtificialViscousCqMultiplier,0.,true);
    const auto& pairs=db.connectivityMap().nodePairList();
    const auto stride=std::max(size_t(1),pairs.size()/1024);
    for(size_t n=0;n<pairs.size();++n) {
      const auto& p=pairs[n];const auto a=p.i_list,b=p.j_list,i=p.i_node,j=p.j_node;
      const auto r=x(a,i)-x(b,j),dv=v(a,i)-v(b,j);
      const auto G=.5*(W.evaluateGradient(r,H(b,j),c(a,i))-W.evaluateGradient(-r,H(a,i),c(b,j)));
      double pi,pj,qi,qj;
      q->QPiij(pi,pj,qi,qj,a,i,b,j,x(a,i),H(a,i),H(a,i)*r,v(a,i),rho(a,i),cs(a,i),
          x(b,j),H(b,j),H(b,j)*r,v(b,j),rho(b,j),cs(b,j),fCl.view(),fCq.view(),g.view());
      const double z=V(a,i)*V(b,j)*dv.dot(G);
      if(entropyThermal) {
        VERIFY2(not compatibleEnergyEvolution() and not evolveTotalEnergy(), "Continuous thermal control requires direct thermal integration");
        const auto force=-V(a,i)*V(b,j)*(P(a,i)+P(b,j)+qi+qj)*G;
        const double power=-dv.dot(force);
        // Limit of Eq61/63 as dt -> 0, evaluated at the current RHS state.
        // No acceleration correction to velocity and no finite-step balancing.
        const double fraction=thermalFraction(entropy(a,i),entropy(b,j),power);
        thermal(a,i)+=fraction*power/m(a,i);
        thermal(b,j)+=(1.-fraction)*power/m(b,j);
      }
      native(a,i)+=(P(b,j)+qj)*z/m(a,i);native(b,j)+=(P(a,i)+qi)*z/m(b,j);
      average(a,i)+=(P(b,j)+.5*(qi+qj))*z/m(a,i);average(b,j)+=(P(a,i)+.5*(qi+qj))*z/m(b,j);
      delta(a,i)+=.5*(qi-qj)*z/m(a,i);delta(b,j)-=.5*(qi-qj)*z/m(b,j);
      if(audit and n%stride==0) {
        const auto force=-V(a,i)*V(b,j)*(P(a,i)+P(b,j)+qi+qj)*G;
        const double row[]={double(rhsCallsThisStep),double(a),double(i),double(b),double(j),G.x(),G.y(),qi,qj,
            P(b,j)*z/m(a,i),qj*z/m(a,i),qi*z/m(b,j),.5*(qi+qj)*z/m(a,i),.5*(qi+qj)*z/m(b,j),
            force.x()/m(a,i),force.y()/m(a,i)};
        heatingPairAudit.insert(heatingPairAudit.end(),std::begin(row),std::end(row));
      }
    }
    for(auto k=0u;k<du.size();++k)for(auto i=0u;i<du[k]->numElements();++i) {
      const double before=du(k,i);
      if(averagedHeating and i<du[k]->numInternalElements())du(k,i)+=delta(k,i);
      if(entropyThermal)du(k,i)=thermal(k,i);
      if(audit) {
        const auto& C=c(k,i);const auto& J=g(k,i);
        const double row[]={double(rhsCallsThisStep),double(k),double(i),double(i<du[k]->numInternalElements()),
            m(k,i),x(k,i).x(),x(k,i).y(),V(k,i),rho(k,i),P(k,i),u(k,i),v(k,i).x(),v(k,i).y(),cs(k,i),
            H(k,i).xx(),H(k,i).xy(),H(k,i).yy(),C[0],C[1],C[2],C[3],C[4],C[5],C[6],C[7],C[8],
            J.xx(),J.xy(),J.yx(),J.yy(),before,du(k,i),native(k,i),average(k,i),acc(k,i).x(),acc(k,i).y(),
            delta(k,i),0.,0.,0.};
        heatingNodeAudit.insert(heatingNodeAudit.end(),std::begin(row),std::end(row));
      }
    }
  }
  if(compatibleEnergyEvolution()) {
    auto acceleration=derivs.fields(HydroFieldNames::hydroAcceleration,Vec::zero());
    auto du=derivs.fields(IncrementState<D,double>::prefix()+HydroFieldNames::specificThermalEnergy,0.);
    for(auto* bc:this->boundaryConditions()) {bc->applyFieldListGhostBoundary(acceleration);bc->applyFieldListGhostBoundary(du);}
    for(auto* bc:this->boundaryConditions())bc->finalizeGhostBoundary();
  }
  if(counts[1].load()==2) {
    const auto q=state.fields(HydroFieldNames::ArtificialViscosityVelocityGradient,Tensor::zero());
    const auto g=derivs.fields(HydroFieldNames::velocityGradient,Tensor::zero());
    const auto x=state.fields(HydroFieldNames::position,Vec::zero());
    for(auto k=0u;k<q.size();++k)for(auto i=0u;i<q[k]->numInternalElements();++i) {
      const auto& a=q(k,i);const auto& b=g(k,i);
      const double row[]={double(rhsCallsThisStep),double(k),double(i),x(k,i).x(),x(k,i).y(),
          a.xx(),a.xy(),a.yx(),a.yy(),b.xx(),b.xy(),b.yx(),b.yy()};
      avGradientAudit.insert(avGradientAudit.end(),std::begin(row),std::end(row));
    }
  }
}

template<> void SpecificThermalEnergyPolicy<D>::update(const std::string& key,State<D>& state,
    StateDerivatives<D>& derivs,const double multiplier,const double t,const double dt) {
  if(freshRK2 and ++energyCallsThisStep==1) {
    // Native SynchronousRK2 calls update for both stages. Preserve the paper's
    // raw predictor, changing only whether the first RHS is fresh or reused.
    this->updateAsIncrement(key,state,derivs,multiplier,t,dt);
    return;
  }
  ++counts[2];
  auto u=state.fields(HydroFieldNames::specificThermalEnergy,0.);
  const auto m=state.fields(HydroFieldNames::mass,0.);
  const auto v=state.fields(HydroFieldNames::velocity,Vec::zero());
  const auto a=derivs.fields(HydroFieldNames::hydroAcceleration,Vec::zero());
  const auto raw=derivs.fields(IncrementState<D,double>::prefix()+HydroFieldNames::specificThermalEnergy,0.);
  const auto& pairA=derivs.get<PairwiseField<D,Vec>>(HydroFieldNames::pairAccelerations);
  const auto& pairs=mDataBasePtr->connectivityMap().nodePairList();
  auto rate=mDataBasePtr->newFluidFieldList(0.,"paper compatible rate");
  VERIFY2(stageEntropy.size()==u.size(),"Missing paper RHS entropy cache");
  for(auto p=0u;p<pairs.size();++p) {
    const auto& q=pairs[p];const auto A=q.i_list,B=q.j_list,i=q.i_node,j=q.j_node;
    const auto vi=v(A,i)+.5*multiplier*a(A,i),vj=v(B,j)+.5*multiplier*a(B,j);
    const double work=(vj-vi).dot(pairA[p]);
    double f=entropyWeighting(stageEntropy.at(A).at(i),stageEntropy.at(B).at(j),work);
    if(dudtPartition) {
      const double sign=(work>0.)-(work<0.);
      const double wi=std::max(std::numeric_limits<double>::epsilon(),raw(A,i)*sign);
      const double wj=std::max(std::numeric_limits<double>::epsilon(),raw(B,j)*sign);
      f=wi/(wi+wj);
    }
    const auto step=counts[1].load();
    if((step==2 or step==50 or step==100 or step==200 or step==400) and p%std::max(size_t(1),pairs.size()/1024)==0) {
      const double instantaneous=(v(B,j)-v(A,i)).dot(pairA[p]);
      const double sign=(work>0.)-(work<0.);
      const double wi=std::max(std::numeric_limits<double>::epsilon(),raw(A,i)*sign);
      const double wj=std::max(std::numeric_limits<double>::epsilon(),raw(B,j)*sign);
      const Vec rhsVi(stageVelocity.at(A).at(2*i),stageVelocity.at(A).at(2*i+1));
      const Vec rhsVj(stageVelocity.at(B).at(2*j),stageVelocity.at(B).at(2*j+1));
      const double rhsEstimatedWork=(rhsVj-rhsVi+.5*multiplier*(a(B,j)-a(A,i))).dot(pairA[p]);
      const double row[]={double(A),double(i),double(B),double(j),work,instantaneous,
          stageEntropy.at(A).at(i),stageEntropy.at(B).at(j),raw(A,i),raw(B,j),
          entropyWeighting(stageEntropy.at(A).at(i),stageEntropy.at(B).at(j),work),
          entropyWeighting(stageEntropy.at(A).at(i),stageEntropy.at(B).at(j),instantaneous),
          wi/(wi+wj),m(A,i),m(B,j),rhsEstimatedWork,double(step)};
      partitionAudit.insert(partitionAudit.end(),std::begin(row),std::end(row));
    }
    rate(A,i)+=f*work;rate(B,j)+=(1.-f)*work*m(A,i)/m(B,j);
  }
  for(auto k=0u;k<u.size();++k)for(auto i=0u;i<u[k]->numInternalElements();++i)u(k,i)+=multiplier*rate(k,i);
}

template<> void SpecificThermalEnergyPolicy<D>::updateAsIncrement(const std::string& key,State<D>& state,
    StateDerivatives<D>& derivs,const double multiplier,const double t,const double dt) {
  ++counts[3];
  std::string field,node;StateBase<D>::splitFieldKey(key,field,node);
  auto u=state.fields(field,0.);IncrementState<D,double> policy;
  for(auto f:u) policy.updateAsIncrement(State<D>::key(*f),state,derivs,multiplier,t,dt);
}
}
