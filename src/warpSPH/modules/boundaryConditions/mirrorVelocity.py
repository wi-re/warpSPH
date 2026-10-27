"""Free-slip reflection of fluid velocity relative to a moving wall."""


def mirrorVelocity(velocity, wallVelocity, normal):
    """Preserve tangential velocity and reverse the relative normal component.

    normal is a unit surface normal. Its sign does not affect the reflection.
    """
    relative = velocity - wallVelocity
    normalVelocity = (relative * normal).sum(-1, keepdim=True) * normal
    return wallVelocity + (relative - normalVelocity) - normalVelocity
