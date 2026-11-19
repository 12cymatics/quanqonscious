#include "rigid-coupler.hpp"
#include <bit>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <string>

using namespace faraday_native;
namespace {
struct Reader {
  std::ifstream stream;
  explicit Reader(const char* path):stream(path,std::ios::binary){if(!stream)throw std::runtime_error("Cannot open prepared native case");}
  std::string bytes(std::size_t n){std::string s(n,'\0');if(!stream.read(s.data(),static_cast<std::streamsize>(n)))throw std::runtime_error("Truncated prepared native case");return s;}
  std::uint64_t integer(std::size_t n){const auto s=bytes(n);std::uint64_t v=0;for(std::size_t i=0;i<n;i++)v|=std::uint64_t(static_cast<unsigned char>(s[i]))<<(8*i);return v;}
  std::uint32_t u32(){return static_cast<std::uint32_t>(integer(4));}
  std::uint64_t u64(){return integer(8);}
  double real(){const double v=std::bit_cast<double>(u64());if(!std::isfinite(v))throw std::runtime_error("Non-finite value in prepared native case");return v;}
};
void quoted(std::ostream& o,const std::string& text){o<<'"';for(unsigned char c:text){if(c=='"'||c=='\\')o<<'\\'<<char(c);else if(c=='\n')o<<"\\n";else if(c=='\r')o<<"\\r";else if(c=='\t')o<<"\\t";else if(c<32)o<<"\\u00"<<std::hex<<std::setw(2)<<std::setfill('0')<<int(c)<<std::dec;else o<<char(c);}o<<'"';}
void vector(std::ostream& o,Vec3 v){o<<'['<<v.x<<','<<v.y<<','<<v.z<<']';}
void array(std::ostream& o,std::span<const double> a){o<<'[';for(std::size_t i=0;i<a.size();i++){if(i)o<<',';if(!std::isfinite(a[i]))throw std::runtime_error("Non-finite native output field");o<<a[i];}o<<']';}
void frame(std::ostream& o,const NativeFluid& f,const RigidCoupler& coupled,std::size_t steps,const CoupledStepReport& report){
  const int keys[]={0,2,1,3,14,20,9,10,11};const char* names[]={"rf","rc","sf","sc","eta","Ht","u","v","w"};
  // Refuse before starting a JSON line, preserving parseable accepted frames.
  for(const int key:keys)for(const double x:f.array(key))if(!std::isfinite(x))throw std::runtime_error("Non-finite native output field");
  for(const auto& g:coupled.grains())g.validate();
  o<<"{\"type\":\"frame\",\"time\":"<<f.time()<<",\"steps\":"<<steps<<",\"initial\":"<<(steps?"false":"true")<<",\"nr\":"<<f.nr()<<",\"nth\":"<<f.nth()<<",\"nz\":"<<f.nz()<<",\"R\":"<<f.config().R<<",\"h\":"<<f.config().h<<",\"contact\":\"pinned\"";
  for(std::size_t i=0;i<9;i++){o<<",\""<<names[i]<<"\":";array(o,f.array(keys[i]));}
  o<<",\"grains\":[";
  for(std::size_t i=0;i<coupled.grains().size();i++){const auto& g=coupled.grains()[i];if(i)o<<',';o<<"{\"id\":"<<g.id<<",\"centre\":";vector(o,g.centre);o<<",\"radius\":"<<g.radius<<",\"density\":"<<g.density<<",\"velocity\":";vector(o,g.velocity);o<<",\"omega\":";vector(o,g.omega);o<<'}';}
  o<<"],\"diagnostics\":{\"dt\":"<<report.dt<<",\"slipResidual\":"<<report.slipResidual<<",\"divergence\":"<<report.divergence<<",\"constraintIterations\":"<<report.constraintIterations<<",\"markers\":"<<report.markerCount<<",\"eventRefinements\":"<<report.eventRefinements<<",\"contacts\":"<<report.contacts<<"}}\n";o.flush();if(!o)throw std::runtime_error("Cannot write complete native frame");
}
}

int main(int argc,char** argv){
  std::cout<<std::setprecision(17);std::cerr<<std::setprecision(17);
  try{
    if(argc!=2)throw std::invalid_argument("Usage: faraday-native prepared-case.fns > result.jsonl");
    Reader input(argv[1]);if(input.bytes(8)!="FNSCASE1"||input.u32()!=1)throw std::invalid_argument("Unsupported prepared native case format/version");
    FluidConfig c;const auto readDimension=[&](){const auto n=input.u32();if(n>std::uint32_t(std::numeric_limits<int>::max()))throw std::length_error("Native dimension outside signed index range");return static_cast<int>(n);};
    c.nr=readDimension();c.nth=readDimension();c.nz=readDimension();if(input.u32()!=1)throw std::invalid_argument("Native case must use pinned contact line");
    const auto budget=input.u64();if(budget>std::numeric_limits<std::size_t>::max())throw std::length_error("Resource budget cannot be represented");c.memoryBudgetBytes=static_cast<std::size_t>(budget);
    c.R=input.real();c.h=input.real();c.rho=input.real();c.nu=input.real();c.gamma=input.real();c.g=input.real();c.dtheta=input.real();
    const double omega=input.real(),acceleration=input.real(),duration=input.real(),dt=input.real(),outputInterval=input.real();
    CouplingOptions options;options.friction=input.real();options.restitution=input.real();
    if(!(omega>=0&&duration>0&&dt>0&&dt<=duration&&outputInterval>0&&std::ceil(duration/dt)<=9007199254740991.0&&std::ceil(duration/outputInterval)<=9007199254740991.0))throw std::invalid_argument("Invalid or unrepresentable native drive/time controls");
    const auto grainCount=input.u32(),arrayCount=input.u32(),jsonBytes=input.u32();
    if(arrayCount!=19||jsonBytes>16*1024*1024)throw std::invalid_argument("Invalid native array count or provenance record length");
    if(std::uint64_t(grainCount)*sizeof(RigidGrain)>budget)throw std::length_error("Grain input exceeds memory budget");
    const auto provenance=input.bytes(jsonBytes);std::vector<RigidGrain> grains;grains.reserve(grainCount);
    for(std::uint32_t n=0;n<grainCount;n++){
      RigidGrain g;const auto id=input.u64();if(id>9007199254740991ULL||id>std::numeric_limits<std::size_t>::max())throw std::invalid_argument("Grain id outside portable safe integer range");g.id=static_cast<std::size_t>(id);
      g.centre={input.real(),input.real(),input.real()};g.radius=input.real();g.density=input.real();g.velocity={input.real(),input.real(),input.real()};g.omega={input.real(),input.real(),input.real()};g.validate();grains.push_back(g);
    }
    NativeFluid fluid(c);std::set<std::uint32_t> keys;
    const std::set<std::uint32_t> expected={0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,20,62,63,64};
    for(std::uint32_t a=0;a<arrayCount;a++){
      const auto key=input.u32();if(!expected.count(key)||!keys.insert(key).second)throw std::invalid_argument("Unexpected or repeated native array");
      const auto count=input.u64();auto values=fluid.array(static_cast<int>(key));if(count!=values.size())throw std::invalid_argument("Native array size differs from the canonical grid");for(double& value:values)value=input.real();
    }
    if(input.stream.peek()!=std::char_traits<char>::eof())throw std::invalid_argument("Trailing bytes after native case");
    fluid.initialize();
    if(dt>fluid.stableStep(acceleration))throw std::invalid_argument("Requested dt exceeds the native fluid stability bound; no silent time-step substitution");
    RigidCoupler coupled(fluid,std::move(grains),options);
    std::cerr<<"{\"stage\":\"native-working-arrays\",\"conservativeBytes\":"<<coupled.workingArrayBytes()<<",\"resourceBudgetBytes\":"<<c.memoryBudgetBytes<<"}\n";
    std::cout<<"{\"type\":\"native-case\",\"format\":\"faraday-native-recording\",\"version\":1,\"caseJson\":";quoted(std::cout,provenance);std::cout<<",\"nativeFluidBytes\":"<<fluid.allocatedBytes()<<"}\n";
    std::size_t steps=0;CoupledStepReport report;frame(std::cout,fluid,coupled,steps,report);double nextOutput=outputInterval;
    while(fluid.time()<duration){
      const double remaining=duration-fluid.time(),step=std::min(dt,remaining);
      if(!(step>0&&fluid.time()+step>fluid.time()))throw std::runtime_error("Physical time step no longer advances representably");
      if(step>fluid.stableStep(acceleration))throw std::runtime_error("Current flow requires a smaller time step; last accepted frame retained");
      report=coupled.step(step,c.g+acceleration*std::cos(omega*fluid.time()));steps++;
      std::cerr<<"{\"steps\":"<<steps<<",\"time\":"<<fluid.time()<<",\"dt\":"<<report.dt<<",\"slip\":"<<report.slipResidual<<",\"divergence\":"<<report.divergence<<",\"constraintIterations\":"<<report.constraintIterations<<"}\n";
      if(fluid.time()>=nextOutput||fluid.time()>=duration){
        frame(std::cout,fluid,coupled,steps,report);
        if(fluid.time()<duration){
          nextOutput=(std::floor(fluid.time()/outputInterval)+1)*outputInterval;
          if(!(nextOutput>fluid.time()))throw std::runtime_error("Output interval no longer advances representably");
        }
      }
    }
    std::cout<<"{\"type\":\"complete\",\"time\":"<<fluid.time()<<",\"steps\":"<<steps<<"}\n";return 0;
  }catch(const std::exception& e){std::cerr<<"Native calculation refused: "<<e.what()<<'\n';std::cout<<"{\"type\":\"error\",\"message\":";quoted(std::cout,e.what());std::cout<<"}\n";return 1;}
}
