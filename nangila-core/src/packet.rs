//! Packet Protocol for Gradient Transmission
//!
//! Defines the wire format for compressed gradient packets,
//! including headers for step synchronization, CRC integrity,
//! and desync recovery flags.

use std::io::{Read, Write};

/// Magic bytes for Nangila packets: "NG" (0x4E47)
pub const PACKET_MAGIC: u16 = 0x4E47;

/// Current protocol version
pub const PROTOCOL_VERSION: u8 = 1;

/// Packet flags
pub mod flags {
    /// Force synchronization: send full FP32, reset predictor state
    pub const FORCE_SYNC: u8 = 0x01;
    /// This packet contains Driver layer data
    pub const DRIVER: u8 = 0x02;
    /// This is a Passenger layer (empty payload)
    pub const PASSENGER: u8 = 0x04;
    /// Request acknowledgment
    pub const ACK_REQUEST: u8 = 0x08;
    /// This is an acknowledgment packet
    pub const ACK_RESPONSE: u8 = 0x10;
    /// Desync detected, initiating recovery
    pub const DESYNC_RECOVERY: u8 = 0x20;
}

/// Packet header (16 bytes, fixed size)
///
/// Layout:
/// ```text
/// +--------+--------+--------+--------+
/// | magic (2) | ver | flags |   step (4)   |
/// +--------+--------+--------+--------+
/// |     layer_id (4)     |     crc32 (4)    |
/// +--------+--------+--------+--------+
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct PacketHeader {
    /// Magic bytes (0x4E47 for "NG")
    pub magic: u16,
    /// Protocol version
    pub version: u8,
    /// Flags (see `flags` module)
    pub flags: u8,
    /// Training step counter
    pub step: u32,
    /// Layer identifier
    pub layer_id: u32,
    /// CRC32 of payload (computed after header)
    pub crc32: u32,
}

impl PacketHeader {
    /// Header size in bytes
    pub const SIZE: usize = 16;

    /// Create a new header for driver data
    pub fn new_driver(step: u32, layer_id: u32) -> Self {
        Self {
            magic: PACKET_MAGIC,
            version: PROTOCOL_VERSION,
            flags: flags::DRIVER,
            step,
            layer_id,
            crc32: 0, // Computed later
        }
    }

    /// Create a new header for passenger (empty payload)
    pub fn new_passenger(step: u32, layer_id: u32) -> Self {
        Self {
            magic: PACKET_MAGIC,
            version: PROTOCOL_VERSION,
            flags: flags::PASSENGER,
            step,
            layer_id,
            crc32: 0,
        }
    }

    /// Create a force-sync header
    pub fn new_force_sync(step: u32, layer_id: u32) -> Self {
        Self {
            magic: PACKET_MAGIC,
            version: PROTOCOL_VERSION,
            flags: flags::FORCE_SYNC | flags::DRIVER,
            step,
            layer_id,
            crc32: 0,
        }
    }

    /// Check if this is a valid Nangila packet
    pub fn is_valid(&self) -> bool {
        self.magic == PACKET_MAGIC && self.version == PROTOCOL_VERSION
    }

    /// Check if FORCE_SYNC flag is set
    pub fn is_force_sync(&self) -> bool {
        self.flags & flags::FORCE_SYNC != 0
    }

    /// Check if this is a driver packet
    pub fn is_driver(&self) -> bool {
        self.flags & flags::DRIVER != 0
    }

    /// Check if this is a passenger packet
    pub fn is_passenger(&self) -> bool {
        self.flags & flags::PASSENGER != 0
    }

    /// Check if desync recovery is active
    pub fn is_desync_recovery(&self) -> bool {
        self.flags & flags::DESYNC_RECOVERY != 0
    }

    /// Set the CRC32 value
    pub fn with_crc(mut self, crc: u32) -> Self {
        self.crc32 = crc;
        self
    }

    /// Serialize header to bytes
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut bytes = [0u8; Self::SIZE];
        bytes[0..2].copy_from_slice(&self.magic.to_le_bytes());
        bytes[2] = self.version;
        bytes[3] = self.flags;
        bytes[4..8].copy_from_slice(&self.step.to_le_bytes());
        bytes[8..12].copy_from_slice(&self.layer_id.to_le_bytes());
        bytes[12..16].copy_from_slice(&self.crc32.to_le_bytes());
        bytes
    }

    /// Deserialize header from bytes
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < Self::SIZE {
            return None;
        }

        Some(Self {
            magic: u16::from_le_bytes([bytes[0], bytes[1]]),
            version: bytes[2],
            flags: bytes[3],
            step: u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
            layer_id: u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]),
            crc32: u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]),
        })
    }

    /// Write header to a writer
    pub fn write_to<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_all(&self.to_bytes())
    }

    /// Read header from a reader
    pub fn read_from<R: Read>(reader: &mut R) -> std::io::Result<Self> {
        let mut bytes = [0u8; Self::SIZE];
        reader.read_exact(&mut bytes)?;
        Self::from_bytes(&bytes)
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid header"))
    }
}

/// Compute CRC32 of a byte slice
///
/// Uses the IEEE polynomial (same as Ethernet, gzip, etc.)
pub fn compute_crc32(data: &[u8]) -> u32 {
    // CRC32 lookup table (IEEE polynomial 0xEDB88320)
    const CRC_TABLE: [u32; 256] = {
        let mut table = [0u32; 256];
        let mut i = 0;
        while i < 256 {
            let mut crc = i as u32;
            let mut j = 0;
            while j < 8 {
                if crc & 1 != 0 {
                    crc = (crc >> 1) ^ 0xEDB88320;
                } else {
                    crc >>= 1;
                }
                j += 1;
            }
            table[i] = crc;
            i += 1;
        }
        table
    };

    let mut crc = 0xFFFFFFFFu32;
    for &byte in data {
        let index = ((crc ^ byte as u32) & 0xFF) as usize;
        crc = (crc >> 8) ^ CRC_TABLE[index];
    }
    !crc
}

/// Verify CRC32 of data matches expected value
pub fn verify_crc32(data: &[u8], expected: u32) -> bool {
    compute_crc32(data) == expected
}

/// Full packet (header + payload)
#[derive(Clone, Debug)]
pub struct Packet {
    pub header: PacketHeader,
    pub payload: Vec<u8>,
}

impl Packet {
    /// Create a new packet with computed CRC
    pub fn new(mut header: PacketHeader, payload: Vec<u8>) -> Self {
        header.crc32 = compute_crc32(&payload);
        Self { header, payload }
    }

    /// Verify packet integrity
    pub fn verify(&self) -> bool {
        self.header.is_valid() && compute_crc32(&self.payload) == self.header.crc32
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(PacketHeader::SIZE + self.payload.len());
        bytes.extend_from_slice(&self.header.to_bytes());
        bytes.extend_from_slice(&self.payload);
        bytes
    }

    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < PacketHeader::SIZE {
            return None;
        }

        let header = PacketHeader::from_bytes(&bytes[..PacketHeader::SIZE])?;
        let payload = bytes[PacketHeader::SIZE..].to_vec();

        Some(Self { header, payload })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_roundtrip() {
        let header = PacketHeader::new_driver(12345, 42);
        let bytes = header.to_bytes();
        let recovered = PacketHeader::from_bytes(&bytes).unwrap();
        assert_eq!(header, recovered);
    }

    #[test]
    fn test_packet_crc() {
        let header = PacketHeader::new_driver(100, 5);
        let payload = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let packet = Packet::new(header, payload);

        assert!(packet.verify());
    }

    #[test]
    fn test_crc_detects_corruption() {
        let header = PacketHeader::new_driver(100, 5);
        let payload = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let mut packet = Packet::new(header, payload);

        // Corrupt one byte
        packet.payload[3] ^= 0xFF;

        assert!(!packet.verify());
    }

    #[test]
    fn test_force_sync_flag() {
        let header = PacketHeader::new_force_sync(500, 10);
        assert!(header.is_force_sync());
        assert!(header.is_driver());
    }

    #[test]
    fn test_passenger_flag() {
        let header = PacketHeader::new_passenger(500, 10);
        assert!(header.is_passenger());
        assert!(!header.is_driver());
    }
}
