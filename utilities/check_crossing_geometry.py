import numpy as np

def segments_intersect(p1, p2, p3, p4):
    """Check if segment p1-p2 intersects segment p3-p4 (not including endpoints)"""
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    
    # Check if they actually cross (strict interior intersection)
    ccw1 = ccw(p1, p3, p4)
    ccw2 = ccw(p2, p3, p4)
    ccw3 = ccw(p1, p2, p3)
    ccw4 = ccw(p1, p2, p4)
    
    if ccw1 != ccw2 and ccw3 != ccw4:
        # Compute intersection point
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return False, None
            
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        
        px = x1 + t * (x2 - x1)
        py = y1 + t * (y2 - y1)
        
        return True, (px, py)
    
    return False, None

pts = np.array([
    [0.0, 0.0],  # 0
    [1.0, 0.0],  # 1
    [1.0, 1.0],  # 2
    [0.0, 1.0],  # 3
    [0.5, 0.3],  # 4
    [0.7, 0.6],  # 5
], dtype=float)

# Check edge (2,4) vs edge (1,5)
p2 = pts[2]
p4 = pts[4]
p1 = pts[1]
p5 = pts[5]

print(f"Edge (2,4): {p2} -> {p4}")
print(f"Edge (1,5): {p1} -> {p5}")

intersects, point = segments_intersect(p2, p4, p1, p5)
print(f"\nDo they intersect? {intersects}")
if intersects:
    print(f"Intersection point: {point}")
