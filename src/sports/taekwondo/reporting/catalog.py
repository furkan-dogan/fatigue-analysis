"""Legacy taekwondo report data; rules remain unvalidated."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricInfo:
    label: str
    unit: str
    plain: str
    calculation: str
    interpretation: str
    fatigue_direction: int
    decimals: int = 1


METRICS: dict[str, MetricInfo] = {
    "active_knee_rom_deg": MetricInfo(
        "Diz Hareket Açıklığı",
        "derece",
        "Tekme sırasında aktif bacağın dizinin ne kadar açılıp kapandığını gösterir.",
        "Tekme penceresinde aktif diz açısının maksimum değeri ile minimum değeri arasındaki fark alınır.",
        "Post değeri belirgin düşerse eklem hareket açıklığı azalmış, kas sertliği veya yorgunluk artmış olabilir.",
        -1,
        1,
    ),
    "active_min_knee_angle_deg": MetricInfo(
        "Diz Fleksiyon Açısı",
        "derece",
        "Tekme öncesinde dizin en fazla büküldüğü açıyı gösterir.",
        "Tekme penceresinde aktif diz açısının minimum değeri alınır.",
        "Açı küçüldükçe diz daha fazla bükülmüş kabul edilir; bu değer tekme hazırlık fazını anlamaya yardım eder.",
        -1,
        1,
    ),
    "active_peak_knee_angle_deg": MetricInfo(
        "Diz Ekstansiyon Açısı",
        "derece",
        "Tekme sırasında dizin en fazla açıldığı açıyı gösterir.",
        "Tekme penceresinde aktif diz açısının maksimum değeri alınır.",
        "180 dereceye yaklaşması dizin daha fazla açıldığını gösterir; tekme uzatma fazını anlamaya yardım eder.",
        -1,
        1,
    ),
    "active_peak_knee_vel_deg_s": MetricInfo(
        "Maksimum Diz Hızı",
        "derece/sn",
        "Tekme sırasında dizin ulaştığı en yüksek açısal hızdır.",
        "Diz açısı zaman serisinden açısal hız hesaplanır; tekme penceresindeki mutlak maksimum hız alınır.",
        "Post değeri düşerse patlayıcı hız üretimi azalmış kabul edilir. Yorgunluk için en güçlü göstergelerden biridir.",
        -1,
        0,
    ),
    "active_mean_knee_vel_deg_s": MetricInfo(
        "Ortalama Diz Hızı",
        "derece/sn",
        "Tekme boyunca diz hareketinin ortalama hızını gösterir.",
        "Tekme penceresindeki mutlak diz açısal hızlarının ortalaması alınır.",
        "Düşüş, tüm tekme boyunca hareket temposunun yavaşladığını gösterir.",
        -1,
        0,
    ),
    "time_to_peak_knee_vel_sec": MetricInfo(
        "Peak Hıza Ulaşma Süresi",
        "sn",
        "Tekmenin başlangıcından maksimum diz hızına ulaşana kadar geçen süredir.",
        "Tekme başlangıcı ile tekme penceresindeki maksimum diz hızı zamanı arasındaki farktır.",
        "Post değeri artarsa sporcu maksimum hıza daha geç ulaşıyor demektir; patlayıcı güçte yorgunluk göstergesidir.",
        +1,
        2,
    ),
    "peak_kick_height_norm": MetricInfo(
        "Tekme Yüksekliği",
        "gövde oranı",
        "Ayağın, sporcunun gövde uzunluğuna göre ne kadar yükseldiğini gösterir.",
        "Ayak yüksekliği, sporcunun gövde uzunluğuna oranlanır. Bu yüzden kamera uzaklığı ve boy farkı daha az etkiler.",
        "Post değeri düşerse yorgunlukla kalça fleksiyonu veya teknik yükseklik korunamamış olabilir.",
        -1,
        2,
    ),
    "active_peak_foot_speed_norm": MetricInfo(
        "Ayak Hızı",
        "gövde/sn",
        "Ayağın, sporcunun gövde uzunluğuna göre ne kadar hızlı hareket ettiğini gösterir.",
        "Frame bazlı ayak konumu değişimi FPS ile hıza çevrilir ve gövde uzunluğuna oranlanır.",
        "Post değeri düşerse tekme uç hızında azalma vardır; performans yorgunluğu ile uyumludur.",
        -1,
        2,
    ),
    "duration_sec": MetricInfo(
        "Tekme Süresi",
        "sn",
        "Tekmenin başlangıçtan bitişe kadar sürdüğü toplam zamandır.",
        "Ayak yüksekliği ve diz hızı sinyallerinden bulunan tekme penceresinin süre uzunluğu hesaplanır.",
        "Post değeri artarsa hareket yavaşlamış olabilir; tek başına değil hız ve ROM ile birlikte yorumlanır.",
        +1,
        2,
    ),
    "extension_dur_sec": MetricInfo(
        "Uzatma Süresi",
        "sn",
        "Dizin en kapalı noktadan maksimum uzamaya gittiği patlayıcı fazın süresidir.",
        "Chamber anı ile dizin uzadığı extension anı arasındaki süre hesaplanır.",
        "Uzatma süresinin uzaması patlayıcı fazın yavaşladığını gösterebilir.",
        +1,
        2,
    ),
    "retraction_dur_sec": MetricInfo(
        "Geri Çekim Süresi",
        "sn",
        "Tekme uzatıldıktan sonra bacağın geri toplandığı fazın süresidir.",
        "Extension anı ile tekme bitişi arasındaki süre hesaplanır.",
        "Yorgunlukta geri çekim fazı uzayabilir; savunmaya dönüş gecikir.",
        +1,
        2,
    ),
    "extension_peak_vel_deg_s": MetricInfo(
        "Uzatma Peak Hızı",
        "derece/sn",
        "Tekmenin uzatma fazındaki en yüksek diz açısal hızıdır.",
        "Chamber-extension aralığındaki mutlak diz hızının maksimumu alınır.",
        "Düşüş, vuruş fazındaki patlayıcı kuvvetin azaldığını gösterir.",
        -1,
        0,
    ),
    "retraction_peak_vel_deg_s": MetricInfo(
        "Geri Çekim Peak Hızı",
        "derece/sn",
        "Bacağın geri toplama fazındaki en yüksek diz açısal hızıdır.",
        "Extension-bitiş aralığındaki mutlak diz hızının maksimumu alınır.",
        "Düşüş, tekmeden sonra savunma pozisyonuna dönüşün yavaşladığını gösterir.",
        -1,
        0,
    ),
    "knee_asi": MetricInfo(
        "Diz Asimetri İndeksi",
        "%",
        "Sağ ve sol diz hareket açıklığı arasındaki farkın yüzde karşılığıdır.",
        "ASI = (Sağ ROM - Sol ROM) / iki bacağın ortalaması x 100. Pozitif değer sağ taraf baskınlığını gösterir.",
        "Mutlak değer 10% üstüne çıkarsa literatürde klinik olarak anlamlı asimetri kabul edilir.",
        +1,
        1,
    ),
    "hip_asi": MetricInfo(
        "Kalça Asimetri İndeksi",
        "%",
        "Sağ ve sol kalça hareket açıklığı arasındaki farkın yüzde karşılığıdır.",
        "ASI = (Sağ kalça ROM - Sol kalça ROM) / iki taraf ortalaması x 100.",
        "Mutlak değer 10% üstüne çıkarsa yük dağılımı ve teknik simetri açısından dikkat gerektirir.",
        +1,
        1,
    ),
    "pose_confidence": MetricInfo(
        "Pose Güven Skoru",
        "0-1",
        "Modelin vücut noktalarını ne kadar güvenilir takip ettiğini gösterir.",
        "Tekme penceresindeki landmark güven skorlarının ortalaması alınır.",
        "0.60 altı tekmelerde açı ve hız hesapları gürültülü olabilir; sonuç yorumunda ihtiyatlı kullanılmalıdır.",
        -1,
        2,
    ),
}


PRIMARY_METRICS = [
    "active_peak_knee_vel_deg_s",
    "active_knee_rom_deg",
    "peak_kick_height_norm",
    "active_peak_foot_speed_norm",
    "time_to_peak_knee_vel_sec",
    "duration_sec",
]


EMG_CH1_MUSCLE = "Rectus femoris"


EMG_CH1_GROUP = "Quadriceps / ön uyluk"


EMG_CH2_MUSCLE = "Biceps femoris"


EMG_CH2_GROUP = "Hamstring / arka uyluk"


EMG_MUSCLE_GROUP = f"{EMG_CH1_GROUP} + {EMG_CH2_GROUP}"


EMG_MUSCLE_NAME = f"CH1: {EMG_CH1_MUSCLE} / CH2: {EMG_CH2_MUSCLE}"


EMG_PLACEMENT = (
    "EMG CH1 rectus femoris üzerinde, CH2 biceps femoris üzerinde değerlendirilir. "
    "Rectus femoris tekmede bacağı kaldırma ve diz açma fazını; biceps femoris diz bükme, geri çekme ve frenleme fazını takip eder."
)
