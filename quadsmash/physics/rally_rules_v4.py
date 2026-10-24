"""Single-return rally events under BWF2025, with separate robot safety gates.

All contact booleans/points supplied here must come from native engine forces
and measured geometry. This module never supplies policy observations/forces.
"""
import torch
from .native import segment_plane_crossing
from .court_v4 import top_height


class RallyRulesV4:
    flags = ('launched','in_play','had_contact','legal_initial_contact','string_hit','airborne_string_hit',
             'frame_hit','released','crossed_net','landed_legal_court','landed_target','ground_seen',
             'robot_net_fault','body_ball_fault','wrong_side_fault','double_hit_fault',
             'racket_invasion_fault','body_over_net_fault','under_net_obstruction_fault',
             'engineering_body_crossing','bad_return_fault','net_blocked_fault',
             'shuttle_net_contact','followthrough_crossed','followthrough_finished')
    def __init__(self,count,device,court,radius=.025,dtype=torch.float32):
        self.count,self.device,self.court,self.radius = count,device,court,radius
        for name in self.flags:setattr(self,name,torch.zeros(count,dtype=torch.bool,device=device))
        self.contact_count = torch.zeros(count,dtype=torch.long,device=device)
        self.quiet_frames = torch.zeros_like(self.contact_count)
        for name in ('contact_time','release_time','crossing_time','landing_time','net_margin','launch_time'):
            setattr(self,name,torch.full((count,),float('nan'),dtype=dtype,device=device))
        for name in ('contact_point','outgoing_velocity','crossing_point','landing_point','last_touch_velocity'):
            setattr(self,name,torch.full((count,3),float('nan'),dtype=dtype,device=device))

    def reset(self,ids):
        for name in self.flags:getattr(self,name)[ids]=False
        self.contact_count[ids]=self.quiet_frames[ids]=0
        for name in ('contact_time','release_time','crossing_time','landing_time','net_margin','launch_time',
                     'contact_point','outgoing_velocity','crossing_point','landing_point','last_touch_velocity'):
            getattr(self,name)[ids]=float('nan')

    def launch(self,ids,time):
        self.launched[ids]=self.in_play[ids]=True
        self.launch_time[ids]=time[ids] if torch.is_tensor(time) and time.ndim else time

    def faulted(self):
        result = torch.zeros(self.count,dtype=torch.bool,device=self.device)
        for name in ('robot_net_fault','body_ball_fault','wrong_side_fault','double_hit_fault',
                     'racket_invasion_fault','body_over_net_fault','under_net_obstruction_fault',
                     'engineering_body_crossing','bad_return_fault','net_blocked_fault'):
            result |= getattr(self,name)
        return result

    def advance(self,*,previous_position,position,velocity,time,dt,racket_contact_event,
                racket_touching,separated,contact_point,string_contact,frame_contact,airborne,
                ground_touch,robot_net_touch,body_ball_touch,shuttle_net_touch,
                racket_over_net,body_over_net,engineering_body_crossing,
                under_net_obstruction=None,same_stroke_frame_string=None):
        active = self.in_play.clone()
        zeros = torch.zeros_like(active)
        obstruction = zeros if under_net_obstruction is None else under_net_obstruction
        exception = zeros if same_stroke_frame_string is None else same_stroke_frame_string
        self.robot_net_fault |= active & robot_net_touch
        self.body_ball_fault |= active & body_ball_touch
        self.body_over_net_fault |= active & body_over_net
        self.under_net_obstruction_fault |= active & obstruction
        self.engineering_body_crossing |= active & engineering_body_crossing
        event = active & racket_contact_event
        first = event & (~self.had_contact)
        own_side = torch.isfinite(contact_point).all(-1) & (contact_point[:,0] < self.court['net_plane_x_m'])
        self.wrong_side_fault |= first & (~own_side)
        self.legal_initial_contact |= first & own_side & (~self.faulted())
        self.double_hit_fault |= event & self.had_contact & (~exception)
        self.contact_count += event.long()
        self.contact_point[first]=contact_point[first]
        self.contact_time[first]=time[first]
        self.had_contact |= event
        self.frame_hit |= active & racket_touching & frame_contact
        self.string_hit |= active & racket_touching & string_contact
        self.airborne_string_hit |= active & racket_touching & string_contact & airborne
        # Only the racket may follow through. Returning fully to the own side
        # ends that exception; a later separate crossing is not the same stroke.
        self.racket_invasion_fault |= active & racket_over_net & ((~self.legal_initial_contact)|self.followthrough_finished)
        returned = active & self.followthrough_crossed & (~racket_over_net)
        self.followthrough_finished |= returned
        self.followthrough_crossed |= active & racket_over_net & self.legal_initial_contact
        touched = active & racket_touching
        self.last_touch_velocity[touched]=velocity[touched]
        self.quiet_frames = torch.where(active & self.had_contact & separated,
                                        self.quiet_frames+1,torch.zeros_like(self.quiet_frames))
        release = active & self.had_contact & (~self.released) & (self.quiet_frames>=2)
        self.outgoing_velocity[release]=self.last_touch_velocity[release]
        self.release_time[release]=time[release]
        self.released |= release
        self.bad_return_fault |= release & (self.outgoing_velocity[:,0]<=0)
        self.shuttle_net_contact |= active & shuttle_net_touch
        own = position[:,0]<self.court['net_plane_x_m']
        # Net touch itself is legal. Reject a measured blocked/falling return
        # on the striker's side, including incoming shots before our contact.
        returning_to_striker = torch.where(self.had_contact,own & (velocity[:,0]<=0),
                                           (~own) & (velocity[:,0]>=0))
        blocked = active & self.shuttle_net_contact & returning_to_striker & (velocity[:,2]<0)
        self.net_blocked_fault |= blocked
        crossing,fraction,point = segment_plane_crossing(previous_position,position,0,self.court['net_plane_x_m'])
        first_cross = active & crossing & self.released & torch.isnan(self.crossing_time)
        self.crossing_time[first_cross]=time[first_cross]-dt+fraction[first_cross]*dt
        self.crossing_point[first_cross]=point[first_cross]
        height = top_height(point[:,1],self.court)
        self.net_margin[first_cross]=(point[:,2]-self.radius-height)[first_cross]
        # Actual solid-band collision governs clearance. Touching the tape is
        # allowed; center must pass above its top, not below/around the net.
        self.crossed_net |= (first_cross & self.legal_initial_contact & self.string_hit
                             & (point[:,2]>=height) & (point[:,1].abs()<=self.court['net_width_m']/2)
                             & (~self.faulted()))
        landing,fraction,point = segment_plane_crossing(previous_position,position,2,self.radius,-1)
        first_landing = active & (landing|ground_touch) & (~self.ground_seen)
        point = torch.where((landing|(~ground_touch))[:,None],point,position)
        self.landing_point[first_landing]=point[first_landing]
        self.landing_time[first_landing]=torch.where(landing,time-dt+fraction*dt,time)[first_landing]
        low,high = self.court['opponent_x_limits_m']
        legal_court = (point[:,0]>=low)&(point[:,0]<=high)&(point[:,1].abs()<=self.court['singles_width_m']/2)
        target_low,target_high = self.court['target_x_limits_m']
        target = (point[:,0]>=target_low)&(point[:,0]<=target_high)&(point[:,1].abs()<=self.court['target_y_half_width_m'])
        valid = first_landing & self.crossed_net & (~self.faulted())
        self.landed_legal_court |= valid & legal_court
        self.landed_target |= valid & legal_court & target
        self.ground_seen |= first_landing
        self.in_play &= (~self.faulted()) & (~first_landing)

    def attack(self):
        speed = self.outgoing_velocity.norm(dim=-1)
        return (self.released & self.legal_initial_contact & self.airborne_string_hit
                & (speed>=8.) & (self.outgoing_velocity[:,0]>0) & (self.outgoing_velocity[:,2]<=-1.))

    def complete(self,recovery):
        return self.attack() & self.landed_target & recovery & (~self.faulted())
