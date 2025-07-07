#!/usr/bin/env python3

"""
Distance vs Latency Calculator for Metro WAN Systems

Calculates round trip latency based on distance and finds distance caps 
for specific latency targets like 15ms.

Physical constants:
- Speed of light in vacuum: 299,792,458 m/s
- Speed of light in fiber: ~200,000 km/s (66.7% of vacuum speed)
- Round trip = 2x distance
"""

import math
import argparse

class DistanceLatencyCalculator:
    def __init__(self):
        # Physical constants
        self.speed_of_light_vacuum_km_s = 299792.458  # km/s
        self.fiber_refractive_index = 1.5  # Typical for fiber optic cable
        self.speed_of_light_fiber_km_s = self.speed_of_light_vacuum_km_s / self.fiber_refractive_index
        
        # More accurate: ~200,000 km/s is commonly used
        self.speed_of_light_fiber_km_s = 200000.0  # km/s (2/3 of vacuum speed)
        
    def calculate_round_trip_latency_us(self, distance_km: float) -> float:
        """Calculate round trip latency in microseconds"""
        round_trip_distance_km = distance_km * 2
        latency_seconds = round_trip_distance_km / self.speed_of_light_fiber_km_s
        return latency_seconds * 1000000  # Convert to microseconds
    
    def calculate_round_trip_latency_ms(self, distance_km: float) -> float:
        """Calculate round trip latency in milliseconds"""
        return self.calculate_round_trip_latency_us(distance_km) / 1000
    
    def find_distance_for_latency_target(self, target_latency_ms: float) -> float:
        """Find maximum distance for a given latency target"""
        target_latency_seconds = target_latency_ms / 1000
        round_trip_distance_km = target_latency_seconds * self.speed_of_light_fiber_km_s
        return round_trip_distance_km / 2  # Divide by 2 for one-way distance
    
    def print_current_test_settings(self):
        """Show latency for current test settings"""
        print("🧪 CURRENT TEST SETTINGS ANALYSIS")
        print("=" * 50)
        
        test_distances = [
            ("Standard Metro WAN", 50),
            ("City-Scale Metro", 100), 
            ("Speed-of-Light Test", 150),
            ("Your 1ms Budget", None)  # Special case
        ]
        
        for name, distance in test_distances:
            if distance:
                latency_us = self.calculate_round_trip_latency_us(distance)
                latency_ms = latency_us / 1000
                print(f"📡 {name}: {distance}km radius")
                print(f"   Round trip: {distance*2}km = {latency_us:.1f}μs = {latency_ms:.2f}ms")
                
                # Check against 1ms processing budget
                budget_us = 1000  # 1ms = 1000μs
                if latency_us <= budget_us:
                    status = "✅ WITHIN BUDGET"
                elif latency_us <= budget_us * 2:
                    status = "⚠️ CLOSE TO BUDGET"
                else:
                    status = "❌ EXCEEDS BUDGET"
                print(f"   vs 1ms budget: {status}")
                print()
        
        # Special case: what distance fits in 1ms budget?
        budget_distance = self.find_distance_for_latency_target(1.0)
        print(f"🎯 1ms Processing Budget Limit:")
        print(f"   Max distance: {budget_distance:.1f}km radius ({budget_distance*2:.1f}km round trip)")
        print(f"   Speed of light: {self.speed_of_light_fiber_km_s:,.0f} km/s in fiber")
    
    def print_15ms_analysis(self):
        """Analyze the 15ms target specifically"""
        print("\n🎯 15ms ROUND TRIP LATENCY TARGET")
        print("=" * 50)
        
        target_ms = 15.0
        max_distance = self.find_distance_for_latency_target(target_ms)
        
        print(f"📏 Distance Cap for 15ms Round Trip:")
        print(f"   Maximum radius: {max_distance:.0f} km")
        print(f"   Round trip distance: {max_distance*2:.0f} km")
        print(f"   Actual latency: {self.calculate_round_trip_latency_ms(max_distance):.2f} ms")
        
        # Show some real-world examples
        real_world_examples = [
            ("New York to Chicago", 790),
            ("London to Berlin", 930),
            ("Los Angeles to San Francisco", 380),
            ("Tokyo to Osaka", 400),
            ("Sydney to Melbourne", 710),
            ("Continental US (coast to coast)", 2500),
            ("New York to London", 5500)
        ]
        
        print(f"\n🌍 Real-World Distance Examples:")
        for name, distance in real_world_examples:
            latency = self.calculate_round_trip_latency_ms(distance)
            if latency <= 15.0:
                status = "✅ WITHIN 15ms"
            elif latency <= 20.0:
                status = "⚠️ CLOSE (under 20ms)"
            else:
                status = "❌ EXCEEDS 15ms"
            
            print(f"   {name}: {distance}km = {latency:.1f}ms {status}")
    
    def print_comprehensive_table(self):
        """Print a comprehensive distance vs latency table"""
        print(f"\n📊 DISTANCE vs LATENCY TABLE")
        print("=" * 60)
        print(f"{'Distance (km)':<15} {'Round Trip':<12} {'Latency (μs)':<12} {'Latency (ms)':<12} {'Status'}")
        print("-" * 60)
        
        # Test various distances
        test_distances = [10, 25, 50, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000, 3000]
        
        for distance in test_distances:
            latency_us = self.calculate_round_trip_latency_us(distance)
            latency_ms = latency_us / 1000
            round_trip = distance * 2
            
            # Status based on common targets
            if latency_ms <= 1.0:
                status = "🌟 Sub-1ms"
            elif latency_ms <= 5.0:
                status = "⚡ Sub-5ms"
            elif latency_ms <= 15.0:
                status = "✅ Sub-15ms"
            elif latency_ms <= 50.0:
                status = "⚠️ Regional"
            else:
                status = "❌ Long Distance"
            
            print(f"{distance:<15} {round_trip:<12} {latency_us:<12.0f} {latency_ms:<12.1f} {status}")
    
    def print_physics_limits(self):
        """Show the fundamental physics limits"""
        print(f"\n⚡ FUNDAMENTAL PHYSICS LIMITS")
        print("=" * 50)
        
        print(f"🔬 Physical Constants:")
        print(f"   Speed of light (vacuum): {self.speed_of_light_vacuum_km_s:,.0f} km/s")
        print(f"   Speed of light (fiber): {self.speed_of_light_fiber_km_s:,.0f} km/s")
        print(f"   Fiber refractive index: {self.fiber_refractive_index}")
        print(f"   Speed reduction: {(1 - self.speed_of_light_fiber_km_s/self.speed_of_light_vacuum_km_s)*100:.1f}%")
        
        print(f"\n🎯 Your Requirements Analysis:")
        budget_distance = self.find_distance_for_latency_target(1.0)
        target_distance = self.find_distance_for_latency_target(15.0)
        
        print(f"   1ms processing budget → {budget_distance:.0f}km max radius")
        print(f"   15ms round trip target → {target_distance:.0f}km max radius")
        
        print(f"\n💡 Network Design Implications:")
        if target_distance > 1000:
            print(f"   ✅ 15ms allows continental-scale networks")
        if target_distance > 500:
            print(f"   ✅ 15ms covers major metropolitan regions")
        if budget_distance < 200:
            print(f"   ⚠️ 1ms budget limits to metro/regional networks")

def main():
    parser = argparse.ArgumentParser(description="Distance vs Latency Calculator")
    parser.add_argument("--target-ms", type=float, help="Calculate max distance for target latency (ms)")
    parser.add_argument("--distance", type=float, help="Calculate latency for specific distance (km)")
    parser.add_argument("--table", action="store_true", help="Show comprehensive distance/latency table")
    parser.add_argument("--physics", action="store_true", help="Show physics limits analysis")
    
    args = parser.parse_args()
    
    calc = DistanceLatencyCalculator()
    
    if args.target_ms:
        distance = calc.find_distance_for_latency_target(args.target_ms)
        latency_actual = calc.calculate_round_trip_latency_ms(distance)
        print(f"🎯 Target: {args.target_ms}ms round trip latency")
        print(f"📏 Maximum distance: {distance:.1f}km radius ({distance*2:.1f}km round trip)")
        print(f"✅ Actual latency: {latency_actual:.2f}ms")
        
    elif args.distance:
        latency_us = calc.calculate_round_trip_latency_us(args.distance)
        latency_ms = latency_us / 1000
        print(f"📏 Distance: {args.distance}km radius ({args.distance*2}km round trip)")
        print(f"⚡ Round trip latency: {latency_us:.1f}μs = {latency_ms:.2f}ms")
        
    else:
        # Show full analysis
        calc.print_current_test_settings()
        calc.print_15ms_analysis()
        
        if args.table:
            calc.print_comprehensive_table()
        
        if args.physics:
            calc.print_physics_limits()

if __name__ == "__main__":
    main() 